"""
Quick Start Example - Hazardous Scene Analysis
===============================================
Uses OWLv2 for detection + grounding. Florence kept for captioning only.

Usage: python eu.py
"""

from hd import HazardousSceneAnalyzer
from owl import OWLv2Detector
import json
import time

# ── Initialize ───────────────────────────────────────────────────────────────
t0 = time.time()

# Florence-2 for captioning + LLM for reasoning (unchanged)
analyzer = HazardousSceneAnalyzer(
    model_name="microsoft/Florence-2-base",
    llm_name="meta-llama/Llama-3.2-3B-Instruct"
)

# OWLv2 replaces both YOLO-World AND the separate fire specialist
# — it handles fire/smoke natively via natural language queries
owl = OWLv2Detector(
    model_name="google/owlv2-large-patch14-ensemble",  # most accurate
    # model_name="google/owlv2-base-patch16-ensemble", # use this if VRAM < 6GB
    conf_threshold=0.10,       # 0.05–0.10 for safety scenes (prefer recall)
    iou_dedup_threshold=0.60,  # suppress duplicate boxes of the same hazard type
)

# Patch the two detection methods — everything else in analyzer is unchanged
analyzer.detect_objects = lambda img: owl.detect_objects(img)
analyzer.detect_hazards_by_grounding = lambda img, cap="": owl.detect_hazards_by_grounding(img, cap)

print(f"\n⏱ Model loading: {time.time() - t0:.2f}s")

# ── Analyze ──────────────────────────────────────────────────────────────────
image_path = "/home/lahiru/vision_sem7/Test/RA/3.png"
t1 = time.time()
result = analyzer.analyze(image_path)
print(f"⏱ Analysis latency: {time.time() - t1:.2f}s")
print(f"⏱ Total (load + analyze): {time.time() - t0:.2f}s")

# ── Report ────────────────────────────────────────────────────────────────────
analyzer.print_report(result)

print("\n--- Programmatic Access ---")
print(f"Objects Detected : {result['objects_detected']}")
print(f"Objects (detail) : {[obj['label'] for obj in result['objects_detected_detail']]}")
print(f"Hazard Regions   : {[h['label'] for h in result['hazards_detected_detail']]}")

ss = result["scene_summary"]
print(f"Possible Hazards : {ss['possible_hazards']}")
print(f"Severity         : {ss['severity'].upper()}")
print(f"Explanation      : {result['explanation']}")
print(f"Confidence       : {result['confidence']}")
print(f"Clarifying Q     : {result['clarifying_question']}")

# ── Save ──────────────────────────────────────────────────────────────────────
analyzer.visualize(image_path, result, save_path="annotated_output.jpg")
with open("result.json", "w") as f:
    json.dump(result, f, indent=2)

owl.cleanup()
analyzer.cleanup()

print("\n✅ Done! Check 'annotated_output.jpg' and 'result.json'")