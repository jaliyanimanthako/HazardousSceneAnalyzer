"""
LLMEngine — LLM reasoning wrapper
==================================
Handles: model loading, tokenizer, real-time streaming, and robust JSON parsing.
Isolated here so you can debug/test LLM calls independently of the vision pipeline.
"""

import gc
import json
import re
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Optional


# ── Streaming helper ──────────────────────────────────────────────────────────

class _CaptureStreamer:
    """
    Custom streamer that prints tokens in real-time AND captures the full text.
    Attach to model.generate() via the `streamer` argument.
    """

    def __init__(self, tokenizer, skip_prompt: bool = True):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.text = ""
        self.prompt_done = False
        self.prefix = "  │ "

    def put(self, value):
        if self.skip_prompt and not self.prompt_done:
            self.prompt_done = True
            return
        chunk = self.tokenizer.decode(
            value[0] if value.dim() > 1 else value, skip_special_tokens=True
        )
        self.text += chunk
        for char in chunk:
            sys.stdout.write('\n' + self.prefix if char == '\n' else char)
        sys.stdout.flush()

    def end(self):
        pass

    def get_text(self) -> str:
        return self.text


# ── Engine ────────────────────────────────────────────────────────────────────

class LLMEngine:
    """
    Self-contained wrapper for a causal LLM loaded in 4-bit (BitsAndBytes).

    Debug tips:
        engine   = LLMEngine("meta-llama/Llama-3.2-3B-Instruct")
        response = engine.query(system_prompt="You are helpful.", user_message="Hello")
        data     = LLMEngine.parse_json(response)
    """

    def __init__(self, model_name: str, device: Optional[str] = None):
        self.model_name = model_name
        # device is informational; actual placement is handled by device_map="auto"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading LLM: {model_name}...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        print("   LLM loaded")

    # ── Inference ─────────────────────────────────────────────────────────────

    def query(self,
              system_prompt: str,
              user_message: str,
              stream: bool = True,
              max_new_tokens: int = 1200) -> str:
        """
        Run the LLM with a system + user message pair.
        Returns the generated text (streamed to stdout if stream=True).
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ]
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        if stream:
            streamer = _CaptureStreamer(self.tokenizer)
            print("\n  ┌─ LLM Reasoning ────────────────────────────────────────")
            with torch.no_grad():
                self.model.generate(**gen_kwargs, streamer=streamer)
            print("\n  └─────────────────────────────────────────────────────────")
            return streamer.get_text()

        with torch.no_grad():
            output_ids = self.model.generate(**gen_kwargs)
        return self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

    # ── JSON parsing ──────────────────────────────────────────────────────────

    @staticmethod
    def parse_json(response: str) -> dict:
        """
        Robustly extract and parse JSON from LLM output.

        Handles:
          - Markdown code fences (```json ... ```)
          - Leading / trailing prose
          - Truncated JSON structures (closes open braces/brackets before raising)
        """
        clean = re.sub(r'^```json?\s*|\s*```$', '', response.strip())

        json_match = re.search(r'\{[\s\S]*\}', clean)
        if not json_match:
            raise ValueError("No JSON object found in LLM output")

        candidate = json_match.group()

        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # Attempt structural repair for truncated responses
            open_braces   = candidate.count('{') - candidate.count('}')
            open_brackets = candidate.count('[') - candidate.count(']')
            repaired = candidate + ']' * max(0, open_brackets) + '}' * max(0, open_braces)
            return json.loads(repaired)

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def cleanup(self):
        """Free GPU memory held by this engine."""
        for attr in ("model", "tokenizer"):
            if hasattr(self, attr):
                delattr(self, attr)
        torch.cuda.empty_cache()
        gc.collect()
