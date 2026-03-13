"""
OWLv2 Drop-in Detector — replaces YOLO-World + fire specialist
===============================================================
v2 — Fixes noisy/cluttered output seen in annotated images:
  - Hard per-type instance cap (e.g. max 2 spill boxes, max 3 barrels)
  - Stricter IoU dedup (0.35 instead of 0.60) — kills near-duplicate boxes
  - Min bbox area filter (removes tiny junk boxes < 0.8% of image)
  - Separate conf thresholds per hazard category (spill needs higher bar)
  - Score-ranked survival: within each type, only best N boxes kept

Replaces in HazardousSceneAnalyzer:
  - detect_objects()
  - detect_hazards_by_grounding()

Florence-2 kept for captioning only (get_detailed_caption, get_dense_regions).
INSTALL: pip install transformers torch  (no new packages needed)
"""

import torch
import gc
from PIL import Image
from typing import Optional
from pathlib import Path
import yaml
from transformers import Owlv2Processor, Owlv2ForObjectDetection


def _load_detector_config(config_path: Optional[Path] = None) -> dict:
    cfg_path = config_path or Path(__file__).parent.parent / "configs" / "config.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    required = [
        "owl_hazard_queries",
        "query_to_display",
        "type_max_instances",
        "category_conf",
    ]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise KeyError(f"Missing required keys in {cfg_path.name}: {missing}")

    return cfg


_DETECTOR_CFG = _load_detector_config()
OWL_HAZARD_QUERIES = _DETECTOR_CFG["owl_hazard_queries"]
QUERY_TO_DISPLAY = _DETECTOR_CFG["query_to_display"]
TYPE_MAX_INSTANCES = _DETECTOR_CFG["type_max_instances"]
CATEGORY_CONF = _DETECTOR_CFG["category_conf"]

QUERY_TO_OBJECT_LABEL = {
    q: q.split(" or ")[0].split(" and ")[0] for q in OWL_HAZARD_QUERIES
}


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _iou(b1, b2) -> float:
    xi1, yi1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    xi2, yi2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0


def _area(b) -> float:
    return max(0, b[2]-b[0]) * max(0, b[3]-b[1])


# ── Detector ──────────────────────────────────────────────────────────────────

class OWLv2Detector:
    """
    OWLv2 open-vocabulary hazard detector with three-stage post-processing
    to produce clean, readable annotated images.
    """

    def __init__(self,
                 model_name: str = "google/owlv2-large-patch14-ensemble",
                 conf_threshold: float = 0.14,
                 iou_dedup_threshold: float = 0.35,
                 min_area_fraction: float = 0.008,
                 device: Optional[str] = None):
        """
        Args:
            model_name:          HuggingFace checkpoint:
                                   "google/owlv2-base-patch16-ensemble"  fast, ~4GB VRAM
                                   "google/owlv2-large-patch14-ensemble" best, ~8GB VRAM
            conf_threshold:      Global fallback threshold. CATEGORY_CONF overrides per type.
            iou_dedup_threshold: Boxes of the SAME type overlapping above this → merge.
                                 0.35 is strict — good for dense scenes.
            min_area_fraction:   Minimum bbox area as fraction of total image area.
                                 0.008 = 0.8% — removes tiny noise boxes.
            device:              "cuda", "cpu", or None (auto)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.conf_threshold = conf_threshold
        self.iou_dedup_threshold = iou_dedup_threshold
        self.min_area_fraction = min_area_fraction

        print(f"Loading OWLv2: {model_name} on {self.device}...")
        self.processor = Owlv2Processor.from_pretrained(model_name)
        self.model = Owlv2ForObjectDetection.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device).eval()
        print(f"   OWLv2 loaded ({len(OWL_HAZARD_QUERIES)} queries | "
              f"iou_dedup={iou_dedup_threshold} | min_area={min_area_fraction:.1%})")

    # ── Core inference ────────────────────────────────────────────────────────

    def _run_owlv2(self, image: Image.Image) -> list:
        """Single forward pass. Returns raw detections above 0.04 that pass area filter."""
        inputs = self.processor(
            text=[OWL_HAZARD_QUERIES], images=image, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([[image.height, image.width]], device=self.device)
        results = self.processor.post_process_object_detection(
            outputs, threshold=0.04, target_sizes=target_sizes
        )[0]

        img_area = image.width * image.height
        min_px = self.min_area_fraction * img_area

        detections = []
        for score, label_idx, box in zip(results["scores"], results["labels"], results["boxes"]):
            bbox = box.cpu().tolist()
            if _area(bbox) >= min_px:
                detections.append({
                    "query": OWL_HAZARD_QUERIES[label_idx.item()],
                    "bbox":  bbox,
                    "score": float(score.cpu()),
                })
        return detections

    def _semantic_group(self, query: str) -> str:
        """First word of the display label → semantic group key for dedup/caps."""
        display = QUERY_TO_DISPLAY.get(query, query)
        return display.replace("⚠ ", "").split("/")[0].split(" ")[0].lower()

    def _filter_and_dedup(self, detections: list) -> list:
        """
        Three-stage cleanup pipeline:

        Stage 1 — Per-category confidence threshold
            Spills need score >= 0.22; fire needs only 0.08, etc.
            Cuts category-specific false positives without hurting recall elsewhere.

        Stage 2 — Type-aware IoU dedup at iou_dedup_threshold (default 0.35)
            Highest-scoring box wins. Different types at same location are KEPT.
            (A barrel and a spill can legitimately overlap.)

        Stage 3 — Per-type instance cap
            At most TYPE_MAX_INSTANCES[group] boxes survive per hazard type.
            This is the primary lever for reducing visual clutter.
        """
        # Stage 1: per-category threshold
        filtered = [
            d for d in detections
            if d["score"] >= CATEGORY_CONF.get(
                self._semantic_group(d["query"]),
                CATEGORY_CONF["_default"]
            )
        ]

        # Stage 2: type-aware IoU dedup (best score first)
        filtered.sort(key=lambda d: d["score"], reverse=True)
        kept = []
        for det in filtered:
            group = self._semantic_group(det["query"])
            is_dup = any(
                self._semantic_group(k["query"]) == group
                and _iou(det["bbox"], k["bbox"]) > self.iou_dedup_threshold
                for k in kept
            )
            if not is_dup:
                kept.append(det)

        # Stage 3: per-type instance cap
        type_counts: dict = {}
        final = []
        for det in kept:
            group = self._semantic_group(det["query"])
            cap = TYPE_MAX_INSTANCES.get(group, TYPE_MAX_INSTANCES["_default"])
            if type_counts.get(group, 0) < cap:
                final.append(det)
                type_counts[group] = type_counts.get(group, 0) + 1

        return final

    # ── Public API ────────────────────────────────────────────────────────────

    def detect_objects(self, image: Image.Image) -> dict:
        """DROP-IN for HazardousSceneAnalyzer.detect_objects(). Caches raw detections."""
        print(f"  OWLv2 detection ({len(OWL_HAZARD_QUERIES)} queries)...")
        self._cached_detections = self._filter_and_dedup(self._run_owlv2(image))

        summary = {}
        for d in self._cached_detections:
            g = self._semantic_group(d["query"])
            summary[g] = summary.get(g, 0) + 1
        print(f"     {len(self._cached_detections)} detections: {summary}")

        return {
            "bboxes": [d["bbox"] for d in self._cached_detections],
            "labels": [QUERY_TO_OBJECT_LABEL.get(d["query"], d["query"]) for d in self._cached_detections],
        }

    def detect_hazards_by_grounding(self, image: Image.Image,
                                     caption: str = "") -> dict:
        """DROP-IN for HazardousSceneAnalyzer.detect_hazards_by_grounding().
        Reuses cached detections from detect_objects() — no second forward pass."""
        final = getattr(self, "_cached_detections", None)
        if final is None:
            print(f"  OWLv2 hazard grounding ({len(OWL_HAZARD_QUERIES)} queries, no cache)...")
            final = self._filter_and_dedup(self._run_owlv2(image))
        else:
            print(f"  OWLv2 hazard grounding (reusing cached detections)...")

        return {
            "bboxes": [d["bbox"] for d in final],
            "labels": [QUERY_TO_DISPLAY.get(d["query"], f"⚠ {d['query']}") for d in final],
        }

    def cleanup(self):
        for attr in ("model", "processor"):
            if hasattr(self, attr):
                delattr(self, attr)
        torch.cuda.empty_cache()
        gc.collect()


# ═════════════════════════════════════════════════════════════════════════════
# TUNING GUIDE
# ═════════════════════════════════════════════════════════════════════════════
#
# STILL TOO MANY SPILL BOXES?
#   CATEGORY_CONF["unidentified"] = 0.30     # raise confidence floor
#   TYPE_MAX_INSTANCES["unidentified"] = 1   # hard cap to 1 box
#
# STILL TOO MANY BARREL BOXES?
#   TYPE_MAX_INSTANCES["barrel"] = 2
#   iou_dedup_threshold = 0.25               # even more aggressive merge
#
# FIRE MISSED?
#   CATEGORY_CONF["fire"] = 0.05
#   Add to OWL_HAZARD_QUERIES: "bright orange fire", "burning debris"
#
# OVERALL STILL CLUTTERED?
#   OWLv2Detector(conf_threshold=0.18, iou_dedup_threshold=0.25)
#   TYPE_MAX_INSTANCES["_default"] = 2
#
# MODEL SIZE:
#   base-patch16-ensemble  ~4GB VRAM  faster
#   large-patch14-ensemble ~8GB VRAM  more accurate ← default