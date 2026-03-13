"""
FlorenceEngine — Florence-2 vision model wrapper
================================================
Handles: model loading, image preprocessing, and all Florence-2 inference tasks.
Isolated here so you can debug/test vision independently of the LLM or pipeline.
"""

import gc
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from typing import Optional


class FlorenceEngine:
    """
    Self-contained wrapper for Florence-2.

    Debug tips:
        engine = FlorenceEngine("microsoft/Florence-2-base")
        img    = Image.open("test.png").convert("RGB")
        print(engine.detect_objects(img))
        print(engine.get_detailed_caption(img))
    """

    def __init__(self,
                 model_name: str = "microsoft/Florence-2-base",
                 revision: Optional[str] = None,
                 device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        print(f"Loading Florence-2: {model_name} on {self.device}...")
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
        }
        proc_kwargs = {"trust_remote_code": True}
        if revision:
            model_kwargs["revision"] = revision
            proc_kwargs["revision"] = revision

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, **model_kwargs
        ).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_name, **proc_kwargs)
        print("   Florence-2 loaded")

    # ── Image preprocessing ───────────────────────────────────────────────────

    def preprocess_image(self, image: Image.Image, max_side: int = 1024) -> Image.Image:
        """
        Cap image at max_side on the longest edge.
        Florence-2 degrades significantly on images larger than ~1024px —
        bboxes shift, objects are missed, spatial attention breaks down.
        """
        w, h = image.size
        if max(w, h) <= max_side:
            return image
        scale = max_side / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        print(f"  Resized image {w}x{h} -> {new_w}x{new_h} for Florence")
        return image.resize((new_w, new_h), Image.LANCZOS)

    # ── Core runner ───────────────────────────────────────────────────────────

    def run(self, image: Image.Image, task: str,
            text_input: Optional[str] = None,
            max_new_tokens: int = 512) -> dict:
        """
        Single Florence-2 inference call with:
        - Per-task token budgets
        - Greedy decode (num_beams=1) for stability
        - Explicit attention_mask
        - dtype guard (float16 only on CUDA)
        - Retry wrapper (3 attempts + cache flush on RuntimeError)

        Returns the raw post-processed dict from the processor.
        """
        prompt = task if text_input is None else task + text_input
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        for attempt in range(3):
            try:
                inputs = self.processor(
                    text=prompt, images=image, return_tensors="pt"
                ).to(self.device, dtype)

                with torch.no_grad():
                    generated_ids = self.model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        attention_mask=inputs.get("attention_mask"),
                        max_new_tokens=max_new_tokens,
                        num_beams=1,      # greedy — stable, no OOM from beam search
                        do_sample=False,
                    )

                generated_text = self.processor.batch_decode(
                    generated_ids, skip_special_tokens=False
                )[0]

                return self.processor.post_process_generation(
                    generated_text, task=task, image_size=(image.width, image.height)
                )

            except RuntimeError as e:
                if attempt == 2:
                    raise
                print(f"  Florence attempt {attempt + 1} failed: {e}, retrying...")
                torch.cuda.empty_cache()
                gc.collect()

    # ── Task methods ──────────────────────────────────────────────────────────

    def detect_objects(self, image: Image.Image) -> dict:
        """Detect objects with bounding boxes (OD task)."""
        result = self.run(image, "<OD>", max_new_tokens=256)
        return result.get("<OD>", {"bboxes": [], "labels": []})

    def get_detailed_caption(self, image: Image.Image) -> str:
        """Generate a detailed scene description."""
        result = self.run(image, "<MORE_DETAILED_CAPTION>", max_new_tokens=512)
        return result.get("<MORE_DETAILED_CAPTION>", "")

    def get_dense_regions(self, image: Image.Image) -> dict:
        """Get per-region descriptions across the image."""
        result = self.run(image, "<DENSE_REGION_CAPTION>", max_new_tokens=512)
        return result.get("<DENSE_REGION_CAPTION>", {"bboxes": [], "labels": []})

    def ground_phrase(self, image: Image.Image, phrase: str) -> dict:
        """Ground a natural-language phrase to bounding boxes."""
        result = self.run(
            image, "<CAPTION_TO_PHRASE_GROUNDING>",
            text_input=phrase, max_new_tokens=256
        )
        return result.get("<CAPTION_TO_PHRASE_GROUNDING>", {"bboxes": [], "labels": []})

    def read_text(self, image: Image.Image) -> str:
        """
        Run OCR on the image and return extracted text.
        Useful for reading hazmat placards, warning signs, barrel labels, etc.
        Returns an empty string if no text is found.
        """
        result = self.run(image, "<OCR>", max_new_tokens=256)
        return result.get("<OCR>", "").strip()

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def cleanup(self):
        """Free GPU memory held by this engine."""
        for attr in ("model", "processor"):
            if hasattr(self, attr):
                delattr(self, attr)
        torch.cuda.empty_cache()
        gc.collect()
