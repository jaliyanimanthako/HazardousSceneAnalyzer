"""
Hazardous Scene Analysis — Online VLM runner
=============================================
Uses a single powerful vision-language model (e.g. Qwen2-VL, GPT-4o) to perform
detection, captioning, and hazard assessment in one API call.

This replaces the three-model offline stack (OWLv2 + Florence-2 + Llama) and is
suitable when internet connectivity is available and higher accuracy is preferred.

Supported providers (OpenAI-compatible API):
  Alibaba Qwen  — set QWEN_API_KEY (default provider)
  OpenAI        — set OPENAI_API_KEY and pass --provider openai
  Custom        — pass --base-url and --api-key directly

Usage:
  python online.py <image_path>                        # single image, Qwen default
  python online.py <folder_path>                       # all images in folder
  python online.py <image_path> --provider openai      # use GPT-4o
  python online.py <image_path> --model qwen-vl-max    # override model
  python online.py <image_path> --out-dir /my/results  # custom output directory
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from PIL import Image

from engines.qwen_engine import QwenEngine
from utils.prompts import ONLINE_HAZARD_SYSTEM_PROMPT
from utils.online_processing import extract_bboxes, postprocess, HAZARD_COLORS
import utils.reporting as _reporting

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ── Provider presets ──────────────────────────────────────────────────────────

PROVIDERS = {
    "openrouter": {
        "base_url":   "https://openrouter.ai/api/v1",
        "model":      "qwen/qwen3-vl-8b-instruct",
        "env_key":    "OPENROUTER_API_KEY",
        "api_key":    None,
    },
    "qwen": {
        "base_url":   "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "model":      "qwen-vl-max",
        "env_key":    "QWEN_API_KEY",
        "api_key":    None,
    },
    "openai": {
        "base_url":   None,
        "model":      "gpt-4o",
        "env_key":    "OPENAI_API_KEY",
        "api_key":    None,
    },
}


# ── Single image processing ───────────────────────────────────────────────────

def _process_one(engine: QwenEngine, img_path: Path, out_dir: Path, t0: float) -> dict:
    print(f"\n{'='*60}\n  {img_path.name}\n{'='*60}")

    image        = Image.open(img_path).convert("RGB")
    img_w, img_h = image.size

    t1 = time.time()
    print("  Querying VLM...")
    raw_response = engine.query_image(image, ONLINE_HAZARD_SYSTEM_PROMPT)
    print(f"  VLM latency : {time.time() - t1:.2f}s")
    print(f"  Total so far: {time.time() - t0:.2f}s")

    llm_result                             = QwenEngine.parse_json(raw_response)
    objects_detail, hazards_detail, labels = extract_bboxes(llm_result, img_w, img_h)
    assessment                             = postprocess(llm_result, labels)

    result = {
        "objects_detected":          labels,
        "possible_hazards":          assessment["detected_hazard_types"],
        "severity":                  assessment["severity"],
        "explanation":               assessment["explanation"],
        "confidence":                assessment["confidence"],
        "clarifying_question":       assessment["clarifying_question"],
        "_objects_detected_detail":  objects_detail,
        "_hazards_detected_detail":  hazards_detail,
    }

    _reporting.print_report(result)

    annotated_path = out_dir / f"annotated_{img_path.name}"
    _reporting.visualize(str(img_path), result, HAZARD_COLORS,
                         save_path=str(annotated_path))

    json_path = out_dir / f"{img_path.stem}_result.json"
    clean     = {k: v for k, v in result.items() if not k.startswith("_")}
    with open(json_path, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"  JSON saved  : {json_path}")

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Online hazard analysis using a VLM API (Qwen2-VL, GPT-4o, etc.)"
    )
    parser.add_argument("input", help="Image file or folder path")
    parser.add_argument("--provider", default="openrouter", choices=list(PROVIDERS),
                        help="API provider preset (default: qwen)")
    parser.add_argument("--model",    default=None,
                        help="Override model name (e.g. qwen-vl-max-0809, gpt-4o)")
    parser.add_argument("--base-url", default=None,
                        help="Override API base URL (for custom endpoints)")
    parser.add_argument("--api-key",  default=None,
                        help="API key (defaults to env var for the chosen provider)")
    parser.add_argument("--out-dir",  default="results",
                        help="Output directory (default: results/)")
    return parser.parse_args()


def main():
    args      = _parse_args()
    preset    = PROVIDERS[args.provider]

    api_key   = args.api_key or preset.get("api_key") or os.environ.get(preset["env_key"])
    if not api_key:
        print(f"Error: API key not found. Set {preset['env_key']} env var or pass --api-key.")
        sys.exit(1)

    model_name = args.model    or preset["model"]
    base_url   = args.base_url or preset["base_url"]
    out_dir    = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_path = Path(args.input)

    t0     = time.time()
    engine = QwenEngine(api_key=api_key, model_name=model_name, base_url=base_url)

    # ── Single image ─────────────────────────────────────────────────────────
    if input_path.is_file():
        _process_one(engine, input_path, out_dir, t0)

    # ── Directory ─────────────────────────────────────────────────────────────
    elif input_path.is_dir():
        images = sorted(f for f in input_path.iterdir()
                        if f.suffix.lower() in IMAGE_EXTS)
        if not images:
            print(f"No images found in {input_path}")
            sys.exit(1)

        print(f"\nFound {len(images)} image(s) -> saving to '{out_dir}/'")
        all_results = []
        for img_path in images:
            try:
                result = _process_one(engine, img_path, out_dir, t0)
                clean  = {k: v for k, v in result.items() if not k.startswith("_")}
                all_results.append({"image": img_path.name, **clean})
            except Exception as e:
                print(f"Failed on {img_path.name}: {e}")

        summary_path = out_dir / "all_results.json"
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nCombined results saved to: {summary_path}")

    else:
        print(f"Invalid path: {input_path}")
        sys.exit(1)

    engine.cleanup()
    print(f"\nDone! Results saved in '{out_dir}/'")


if __name__ == "__main__":
    main()
