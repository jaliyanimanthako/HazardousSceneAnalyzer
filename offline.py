"""
Hazardous Scene Analysis — OWLv2 runner
========================================
Uses OWLv2 for detection + grounding. Florence kept for captioning only.

Usage:
  python offline.py <image_path>              # single image
  python offline.py <folder_path>             # all images in folder → results/
  python offline.py <folder_path> <out_dir>   # custom output directory
"""

import json
import sys
import time
from pathlib import Path

from hd import HazardousSceneAnalyzer
from object_Dectetion.owl import OWLv2Detector

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _clean(result: dict) -> dict:
    """Strip internal _-prefixed keys before saving to JSON."""
    return {k: v for k, v in result.items() if not k.startswith("_")}


def _process_one(analyzer, img_path: Path, out_dir: Path, t0: float) -> dict:
    print(f"\n{'='*60}\n📷  {img_path.name}\n{'='*60}")
    t1 = time.time()
    result = analyzer.analyze(str(img_path))
    print(f"⏱  Analysis latency : {time.time() - t1:.2f}s")
    print(f"⏱  Total so far     : {time.time() - t0:.2f}s")

    analyzer.print_report(result)

    annotated_path = out_dir / f"annotated_{img_path.name}"
    analyzer.visualize(str(img_path), result, save_path=str(annotated_path))

    json_path = out_dir / f"{img_path.stem}_result.json"
    with open(json_path, "w") as f:
        json.dump(_clean(result), f, indent=2)
    print(f"  💾 JSON saved : {json_path}")

    return result


def main():
    if len(sys.argv) < 2:
        print("Usage:\n"
              "  python offline.py <image_path>\n"
              "  python offline.py <folder_path> [output_dir]")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    out_dir    = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # ── Load models ──────────────────────────────────────────────────────────
    analyzer = HazardousSceneAnalyzer(
        model_name="microsoft/Florence-2-base",
        llm_name="meta-llama/Llama-3.2-3B-Instruct",
    )
    owl = OWLv2Detector(
        model_name="google/owlv2-large-patch14-ensemble",
        # model_name="google/owlv2-base-patch16-ensemble",  # use if VRAM < 6 GB
        conf_threshold=0.10,
        iou_dedup_threshold=0.60,
    )
    analyzer.detect_objects             = lambda img: owl.detect_objects(img)
    analyzer.detect_hazards_by_grounding = lambda img, cap="": owl.detect_hazards_by_grounding(img, cap)
    print(f"\n⏱  Model loading : {time.time() - t0:.2f}s")

    # ── Single image ─────────────────────────────────────────────────────────
    if input_path.is_file():
        _process_one(analyzer, input_path, out_dir, t0)

    # ── Directory — process all images ───────────────────────────────────────
    elif input_path.is_dir():
        images = sorted(f for f in input_path.iterdir()
                        if f.suffix.lower() in IMAGE_EXTS)
        if not images:
            print(f"❌ No images found in {input_path}")
            sys.exit(1)

        print(f"\n📁 Found {len(images)} image(s) → saving to '{out_dir}/'")
        all_results = []
        for img_path in images:
            try:
                result = _process_one(analyzer, img_path, out_dir, t0)
                all_results.append({"image": img_path.name, **_clean(result)})
            except Exception as e:
                print(f"❌ Failed on {img_path.name}: {e}")

        summary_path = out_dir / "all_results.json"
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n💾 Combined results saved to : {summary_path}")

    else:
        print(f"❌ Invalid path: {input_path}")
        sys.exit(1)

    owl.cleanup()
    analyzer.cleanup()
    print(f"\n✅ Done! Results saved in '{out_dir}/'")


if __name__ == "__main__":
    main()
