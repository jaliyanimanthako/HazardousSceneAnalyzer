# Hazardous Scene Analyzer

A multi-model pipeline for detecting and assessing hazards in industrial environments. It combines open-vocabulary object detection (OWLv2), vision captioning (Florence-2), and LLM-based reasoning (Llama) to produce structured hazard reports with operator-facing explanations.

The entire pipeline runs fully offline — no API calls, no internet connection required at inference time. All models are loaded locally from HuggingFace checkpoints (downloaded once, then cached). This makes it suitable for deployment on onboard compute hardware such as NVIDIA Jetson boards, where network connectivity may be limited or unavailable and inference must happen on the robot itself.

Model choices reflect this constraint: Florence-2-base, OWLv2, and Llama-3.2-3B are compact enough to run on Jetson AGX Orin class hardware (64 GB unified memory) with acceptable latency. Larger variants (Florence-2-large, OWLv2-large, Llama-3.1-8B) can be substituted if more capable hardware is available.

---

## How It Works

The pipeline has four stages:

**1. Detection**
OWLv2 detects objects and hazard regions using 24 natural-language queries (fire, smoke, spills, barrels, structural damage, personnel, etc.). Results go through three cleanup passes: per-category confidence thresholds, type-aware IoU deduplication, and per-type instance caps.

**2. Captioning**
Florence-2 generates a detailed scene description and dense per-region captions. These are used both as input to the LLM and as fallback output if the LLM fails.

**3. Phrase Grounding**
Florence-2's phrase grounding task localises hazard vocabulary against the scene caption. When a hazmat placard is detected, the region is cropped and passed through Florence OCR to identify the UN class number and substance keywords.

**4. Hazard Assessment**
A 4-bit quantized LLM (Llama-3.2-3B-Instruct) receives all evidence — detected objects, scene caption, region descriptions, and grounded hazard locations — and returns a structured JSON assessment. If the LLM fails, a keyword-based fallback runs automatically.

---

## Output

Each image produces:

| Field | Description |
|---|---|
| `objects_detected` | All objects identified by OWLv2 and Florence |
| `possible_hazards` | Canonical hazard types: fire, smoke, chemical, spill, structural, electrical, biological |
| `severity` | Overall severity: low / medium / high / critical |
| `explanation` | Risk-focused narrative for the operator — why the hazards are dangerous and what action is needed |
| `confidence` | Model confidence score (0.0 – 1.0) |
| `clarifying_question` | Single highest-priority question that would change the response protocol |

---

## Project Structure

```
.
├── hd.py                        # Pipeline orchestrator (thin — delegates to modules)
├── offline.py                     # CLI entry point (single image or folder)
│
├── engines/
│   ├── florence_engine.py       # Florence-2 wrapper: loading, preprocessing, all tasks
│   └── llm_engine.py            # LLM wrapper: 4-bit loading, streaming, JSON parsing
│
├── utils/
│   ├── assessment.py            # Hazard assessment logic (LLM + keyword fallback)
│   ├── grounding.py             # Phrase grounding, IoU dedup, placard OCR
│   ├── reporting.py             # Terminal report and annotated image output
│   └── prompts.py               # LLM system prompt and HAZMAT vocabulary
│
├── object_Dectetion/
│   └── owl.py                   # OWLv2 detector — drop-in for Florence detection
│
├── configs/
│   ├── hazard_config.yaml        # Labels, keywords, colours, severity messages, UN hazmat classes
│   └── config.yaml               # OWLv2 queries, confidence thresholds, instance caps
│
├── offline.py                     # Offline runner (OWLv2 + Florence-2 + Llama)
└── online.py                    # Online runner (single VLM API call)
```

---

## Running Modes

### Offline mode — `offline.py`

Uses three local models (OWLv2 + Florence-2 + Llama) running entirely on the device. No internet connection required after the initial model download. Produces colour-coded annotated images with bounding boxes.

```bash
# Single image
python offline.py /path/to/image.jpg

# Entire folder
python offline.py /path/to/folder/

# Custom output directory
python offline.py /path/to/folder/ /path/to/output/
```

### Online mode — `online.py`

Sends the image to a powerful hosted VLM (Qwen2-VL, GPT-4o, or any OpenAI-compatible endpoint) in a single API call. The model performs detection, captioning, and hazard assessment in one shot — no local GPU required. Requires an API key and internet access.

```bash
# Qwen2-VL (default) — set QWEN_API_KEY environment variable
export QWEN_API_KEY=your_key_here
python online.py /path/to/image.jpg

# GPT-4o — set OPENAI_API_KEY
export OPENAI_API_KEY=your_key_here
python online.py /path/to/image.jpg --provider openai

# Custom model or endpoint
python online.py /path/to/folder/ --model qwen-vl-max-0809 --out-dir results/

# Pass API key directly (without env var)
python online.py /path/to/image.jpg --api-key sk-... --provider openai
```

Both modes produce the same JSON output format and use the same terminal report. The online mode does not produce bounding box annotations since the VLM API does not return pixel coordinates.

Output is saved to `results/` (or the specified directory):
- `annotated_<name>.jpg` — annotated image (offline) or original image (online)
- `<name>_result.json` — clean JSON result for that image
- `all_results.json` — combined results for folder runs

### Choosing a mode

| | Offline (`offline.py`) | Online (`online.py`) |
|---|---|---|
| Internet required | No | Yes |
| GPU required | Yes (10–12 GB VRAM) | No |
| Bounding boxes | Yes | No |
| Speed | ~30–50s per image | ~5–15s per image |
| Accuracy | Good | Higher (larger model) |
| Cost | Free after download | API usage cost |

---

## Models

| Model | Role | Default checkpoint | Parameters |
|---|---|---|---|
| OWLv2 | Object / hazard detection | `google/owlv2-large-patch14-ensemble` | 307M |
| Florence-2 | Scene captioning, phrase grounding, OCR | `microsoft/Florence-2-base` | 232M |
| Llama | Hazard reasoning, JSON assessment | `meta-llama/Llama-3.2-3B-Instruct` | 3.21B |

If your GPU has less than 6 GB VRAM, switch OWLv2 to `google/owlv2-base-patch16-ensemble` in `offline.py`.

The LLM is loaded in 4-bit (BitsAndBytes) by default. Florence-2 runs in float16 on CUDA and float32 on CPU.

---

## Offline and Edge Deployment

The pipeline makes no outbound network calls at inference time. After the initial model download, it operates entirely from local weights. This is intentional — the system is designed to run on mobile robots where connectivity cannot be guaranteed.

**Target hardware: NVIDIA Jetson**

The default model selection is sized for Jetson AGX Orin (64 GB unified memory) or similar onboard GPU platforms:

| Model | Approx. memory | Notes |
|---|---|---|
| `owlv2-large-patch14-ensemble` | ~8 GB | Switch to base variant on lower-memory boards |
| `Florence-2-base` | ~1.5 GB | base keeps latency low; large adds ~1 GB |
| `Llama-3.2-3B-Instruct` (4-bit) | ~2 GB | 4-bit quantization via BitsAndBytes |

On Jetson, PyTorch uses the onboard CUDA GPU automatically — no code changes are needed. If BitsAndBytes 4-bit quantization is not available on a given JetPack version, set `use_llm=False` in `HazardousSceneAnalyzer` to fall back to the keyword-based assessment which has no LLM dependency.

**Pre-downloading models for offline use**

Before deploying to a machine without internet access, download the models on a connected machine:

```python
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer

AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base",  trust_remote_code=True)
AutoModelForCausalLM.from_pretrained("google/owlv2-large-patch14-ensemble")
AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
```

HuggingFace caches weights to `~/.cache/huggingface/hub/` by default. Copy that directory to the target machine and set:

```bash
export HF_HUB_OFFLINE=1
```

This prevents any attempt to reach the HuggingFace hub and forces local cache use.

---

## Configuration

All detection behaviour is controlled through YAML — no Python changes needed for tuning.

**`configs/config.yaml`** — OWLv2 settings:
- `queries` — the 24 natural-language detection queries
- `category_conf` — per-type confidence floors (e.g. spill: 0.22, fire: 0.08)
- `type_max_instances` — maximum bounding boxes kept per hazard type

**`configs/hazard_config.yaml`** — Pipeline settings:
- `always_check` — phrases grounded on every image regardless of caption
- `display_labels` — phrase to annotation label mapping
- `substance_hedges` — regex patterns that replace overconfident substance names (e.g. "oil" → "unidentified liquid")
- `hazard_keywords` — keyword fallback lists per hazard type
- `hazmat_classes` — UN class numbers to human-readable names (used for placard OCR)
- `hazmat_placard_keywords` — text keywords found on placards (FLAMMABLE, CORROSIVE, etc.)
- `hazard_colors` — bounding box colours per hazard type

---

## Installation

```bash
pip install -r requirements.txt
```

A CUDA-capable GPU is strongly recommended. The full pipeline (OWLv2-large + Florence-2-base + Llama-3.2-3B) requires approximately 10–12 GB VRAM.

---

## Debugging Individual Modules

Each module is independently testable without running the full pipeline:

```python
# Test Florence-2 on a single image
from engines.florence_engine import FlorenceEngine
from PIL import Image

engine = FlorenceEngine("microsoft/Florence-2-base")
img = Image.open("test.png").convert("RGB")
print(engine.detect_objects(img))
print(engine.get_detailed_caption(img))
print(engine.read_text(img))   # OCR

# Test LLM JSON parsing without GPU
from engines.llm_engine import LLMEngine
raw = '{"hazards": [{"type": "fire"'   # truncated
print(LLMEngine.parse_json(raw))

# Test phrase grounding helpers without models
from utils.grounding import build_scene_phrases
from utils.prompts import HAZMAT_VOCABULARY
phrases = build_scene_phrases("smoke rising from a barrel", HAZMAT_VOCABULARY, ["fire"])
print(phrases)
```
