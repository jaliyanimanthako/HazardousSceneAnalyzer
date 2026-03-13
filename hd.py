"""
HazardousSceneAnalyzer — pipeline orchestrator
===============================================
This file is intentionally thin. All logic lives in dedicated modules:

  engines/florence_engine.py   — vision model (loading, preprocessing, inference)
  engines/llm_engine.py        — reasoning model (loading, streaming, JSON parsing)
  utils/grounding.py           — phrase → bbox detection helpers
  utils/assessment.py          — LLM + keyword hazard assessment
  utils/reporting.py           — visualization and text reporting
  configs/hazard_config.yaml   — all labels, keywords, colours (edit without touching Python)
  utils/prompts.py             — LLM system prompt + HAZMAT_VOCABULARY
  object_Dectetion/owl.py      — OWLv2 drop-in detector
"""

import json
from pathlib import Path
from typing import Union, Optional

import yaml
from PIL import Image

from utils.prompts import HAZARD_SYSTEM_PROMPT, HAZMAT_VOCABULARY
from engines.florence_engine import FlorenceEngine
from engines.llm_engine import LLMEngine
import utils.grounding as _grounding
import utils.assessment as _assessment
import utils.reporting as _reporting

# ── Load config ───────────────────────────────────────────────────────────────
_CFG_PATH = Path(__file__).parent / "configs" / "hazard_config.yaml"
with _CFG_PATH.open("r", encoding="utf-8") as _f:
    _CFG = yaml.safe_load(_f)

# Expose config sections as module-level names for readability
_ALWAYS_CHECK             = _CFG["always_check"]
_DISPLAY_LABELS           = _CFG["display_labels"]
_SUBSTANCE_HEDGES         = [tuple(p) for p in _CFG["substance_hedges"]]
_VALID_HAZARD_TYPES       = set(_CFG["valid_hazard_types"])
_HAZARD_KEYWORDS          = _CFG["hazard_keywords"]
_OD_HAZARD_MAP            = _CFG["od_hazard_map"]
_HAZARD_COLORS            = _CFG["hazard_colors"]
_SEVERITY_MESSAGES        = _CFG["severity_messages"]
_HAZMAT_CLASSES           = _CFG.get("hazmat_classes", {})
_HAZMAT_PLACARD_KEYWORDS  = _CFG.get("hazmat_placard_keywords", {})


# ── Analyzer ──────────────────────────────────────────────────────────────────

class HazardousSceneAnalyzer:
    """
    Orchestrates the full hazard analysis pipeline.

    Internals at a glance:
        self.florence    (FlorenceEngine)  — vision inference
        self.llm_engine  (LLMEngine|None)  — LLM reasoning
        grounding.py                       — phrase → bbox detection
        assessment.py                      — LLM + keyword hazard scoring
        reporting.py                       — annotation + report printing
        hazard_config.yaml                 — all tuneable constants
    """

    def __init__(self,
                 model_name: str = "microsoft/Florence-2-large",
                 llm_name: str = "meta-llama/Llama-3.2-3B-Instruct",
                 use_llm: bool = True,
                 florence_revision: str = None):
        """
        Args:
            model_name:        Florence-2 HuggingFace checkpoint.
            llm_name:          LLM checkpoint for reasoning.
            use_llm:           Set False to skip LLM and use keyword fallback only.
            florence_revision: Optional Florence-2 revision (e.g. "refs/pr/6").
        """
        self.use_llm = use_llm

        self.florence   = FlorenceEngine(model_name, revision=florence_revision)
        self.llm_engine = LLMEngine(llm_name) if use_llm else None

        print("\n✅ All models loaded successfully!\n")

    # ── Vision delegation ─────────────────────────────────────────────────────
    # Thin wrappers kept for backward-compat (euowl.py monkey-patches these).

    def detect_objects(self, image: Image.Image) -> dict:
        return self.florence.detect_objects(image)

    def get_detailed_caption(self, image: Image.Image) -> str:
        return self.florence.get_detailed_caption(image)

    def get_dense_regions(self, image: Image.Image) -> dict:
        return self.florence.get_dense_regions(image)

    def detect_hazards_by_grounding(self, image: Image.Image, caption: str = "") -> dict:
        return _grounding.detect_hazards_by_grounding(
            self.florence, image, caption,
            _DISPLAY_LABELS, HAZMAT_VOCABULARY, _ALWAYS_CHECK,
            hazmat_classes=_HAZMAT_CLASSES,
            hazmat_placard_keywords=_HAZMAT_PLACARD_KEYWORDS,
        )

    # ── Main pipeline ─────────────────────────────────────────────────────────

    def analyze(self, image_input: Union[str, Image.Image]) -> dict:
        """
        Run the full analysis pipeline on one image.

        Args:
            image_input: File path string or PIL Image.

        Returns:
            Result dict with objects, hazards, scene summary, LLM assessment.
        """
        # Load image
        if isinstance(image_input, str):
            image_path = image_input
            image = Image.open(image_path).convert("RGB")
        else:
            image_path = "uploaded_image"
            image = image_input.convert("RGB")

        orig_w, orig_h = image.size

        # Preprocess: cap at 1024px before any Florence call
        image  = self.florence.preprocess_image(image)
        proc_w, proc_h = image.size
        scale_x = float(orig_w) / float(proc_w)
        scale_y = float(orig_h) / float(proc_h)

        def remap_bbox(bbox):
            """Scale bbox coordinates from processed size back to original image size."""
            x1, y1, x2, y2 = [float(v) for v in bbox]
            mapped = [
                int(round(x1 * scale_x)), int(round(y1 * scale_y)),
                int(round(x2 * scale_x)), int(round(y2 * scale_y)),
            ]
            mapped[0] = max(0, min(orig_w, mapped[0]))
            mapped[1] = max(0, min(orig_h, mapped[1]))
            mapped[2] = max(0, min(orig_w, mapped[2]))
            mapped[3] = max(0, min(orig_h, mapped[3]))
            return mapped

        # Step 1 — Object detection
        print("  📦 Detecting objects...")
        objects_result = self.detect_objects(image)

        # Step 2 — Caption + dense regions (sequential: Florence is not thread-safe)
        print("  📝 Generating scene caption...")
        caption = self.get_detailed_caption(image)

        print("  📝 Generating region captions...")
        dense_regions = self.get_dense_regions(image)

        # Step 3 — Phrase grounding + OD cross-reference
        hazard_grounding = self.detect_hazards_by_grounding(image, caption)
        od_hazards       = _grounding.od_results_to_hazards(objects_result, _OD_HAZARD_MAP)
        hazard_grounding = {
            "bboxes": hazard_grounding["bboxes"] + od_hazards["bboxes"],
            "labels": hazard_grounding["labels"] + od_hazards["labels"],
        }

        # Remap all coordinates to original image size
        object_labels = objects_result.get("labels", [])
        bboxes        = [remap_bbox(b) for b in objects_result.get("bboxes", [])]
        hazard_bboxes = [remap_bbox(b) for b in hazard_grounding.get("bboxes", [])]

        dense_region_list = [
            {"description": label, "bbox": remap_bbox(bbox)}
            for label, bbox in zip(
                dense_regions.get("labels", []),
                dense_regions.get("bboxes", [])
            )
        ]
        hazard_grounded_list = [
            {"label": label, "bbox": bbox}
            for label, bbox in zip(hazard_grounding.get("labels", []), hazard_bboxes)
        ]

        # Step 4 — Hazard assessment (LLM or keyword fallback)
        print("  🧠 Analyzing hazards...")
        if self.use_llm and self.llm_engine:
            try:
                hazard_assessment = _assessment.assess_hazards_with_llm(
                    self.llm_engine, HAZARD_SYSTEM_PROMPT,
                    object_labels, caption, dense_region_list, hazard_grounded_list,
                    _SUBSTANCE_HEDGES, _VALID_HAZARD_TYPES,
                )
            except Exception as e:
                print(f"  ⚠ LLM failed ({e}), using keyword fallback...")
                hazard_assessment = _assessment.assess_hazards_keywords(
                    object_labels, caption, _HAZARD_KEYWORDS
                )
        else:
            hazard_assessment = _assessment.assess_hazards_keywords(
                object_labels, caption, _HAZARD_KEYWORDS
            )

        decision_support = hazard_assessment.get("decision_support", "")
        if not decision_support:
            decision_support = _assessment.generate_explanation(
                object_labels, hazard_assessment, _SEVERITY_MESSAGES
            )
        # Deduplicate objects list (OD labels + hazard labels merged)
        objects_all = object_labels + [
            h["label"].replace("⚠ ", "") for h in hazard_grounded_list
        ]
        seen = set()
        objects_deduped = []
        for o in objects_all:
            if o.lower() not in seen:
                seen.add(o.lower())
                objects_deduped.append(o)

        print("  ✅ Analysis complete!")
        return {
            "objects_detected":    objects_deduped,
            "possible_hazards":    hazard_assessment["detected_hazard_types"],
            "severity":            hazard_assessment["severity"],
            "explanation":         decision_support,
            "confidence":          hazard_assessment.get("confidence", 0.5),
            "clarifying_question": hazard_assessment.get("clarifying_question"),
            # Prefixed with _ — used by visualize(), not shown in print_report()
            "_objects_detected_detail": [
                {
                    "label": label,
                    "bounding_box": {"x1": int(b[0]), "y1": int(b[1]),
                                     "x2": int(b[2]), "y2": int(b[3])},
                }
                for label, b in zip(object_labels, bboxes)
            ],
            "_hazards_detected_detail": [
                {
                    "label": label,
                    "bounding_box": {"x1": int(b[0]), "y1": int(b[1]),
                                     "x2": int(b[2]), "y2": int(b[3])},
                }
                for label, b in zip(hazard_grounding.get("labels", []), hazard_bboxes)
            ],
        }

    # ── Reporting (thin wrappers for backward-compat) ─────────────────────────

    def visualize(self, image_input: Union[str, Image.Image],
                  output: dict, save_path: Optional[str] = None) -> Image.Image:
        return _reporting.visualize(image_input, output, _HAZARD_COLORS, save_path)

    def print_report(self, output: dict) -> None:
        _reporting.print_report(output)

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def cleanup(self) -> None:
        """Free GPU memory from both engines."""
        self.florence.cleanup()
        if self.llm_engine:
            self.llm_engine.cleanup()


# ── Batch processing ──────────────────────────────────────────────────────────

def batch_process(image_folder: str, output_folder: str = "./results",
                  use_llm: bool = True) -> list:
    """Process all images in a folder and save annotated results."""
    from tqdm import tqdm

    analyzer      = HazardousSceneAnalyzer(use_llm=use_llm)
    image_folder  = Path(image_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    images     = sorted([f for f in image_folder.iterdir()
                         if f.suffix.lower() in extensions])
    print(f"\n📁 Found {len(images)} images\n")

    results = []
    for img_path in tqdm(images, desc="Processing"):
        try:
            print(f"\n{'='*50}\n📷 {img_path.name}\n{'='*50}")
            result = analyzer.analyze(str(img_path))
            results.append(result)
            analyzer.print_report(result)
            analyzer.visualize(str(img_path), result,
                               str(output_folder / f"annotated_{img_path.name}"))
        except Exception as e:
            print(f"❌ Error processing {img_path.name}: {e}")

    clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    json_path = output_folder / "all_results.json"
    with open(json_path, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"\n💾 Results saved to: {json_path}")
    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:\n  python hd.py <image_path>\n  python hd.py <folder_path> [output_folder]")
        sys.exit(1)

    input_path = Path(sys.argv[1])

    if input_path.is_file():
        analyzer = HazardousSceneAnalyzer()
        result   = analyzer.analyze(str(input_path))
        analyzer.print_report(result)
        analyzer.visualize(str(input_path), result, "annotated_output.jpg")
        with open("result.json", "w") as f:
            json.dump({k: v for k, v in result.items() if not k.startswith("_")}, f, indent=2)
        print("\n✅ Saved: annotated_output.jpg, result.json")
        analyzer.cleanup()

    elif input_path.is_dir():
        batch_process(str(input_path), sys.argv[2] if len(sys.argv) > 2 else "./results")

    else:
        print(f"❌ Invalid path: {input_path}")
