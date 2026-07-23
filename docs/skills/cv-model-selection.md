# CV Model Selection

The CV Model Selection skill covers computer-vision architecture decisions — detector family choice, small-object and high-motion regimes, tracking and re-ID, and the accuracy/latency tradeoffs that actually decide a project.

**Skill directory:** `skills/cv-model-selection/`

## Purpose

Architecture choice is a regime decision, not a leaderboard decision. Two models with identical COCO numbers behave completely differently when objects are eight pixels wide, the camera pans, and the budget is 12 ms per frame. This skill teaches Claude Code to quantify the regime first — object size in pixels, objects per frame, latency budget, dataset size, label noise — and then select a detector, decide whether tracking or a second-stage crop model is required, and reason about how each choice exports to ONNX and TensorRT.

## When to Use

- Picking a detection architecture, or justifying a change to the current one
- Diagnosing collapsed small-object recall (balls, distant players, jersey numbers)
- Deciding whether per-frame detection is sufficient or tracking/temporal modeling is needed
- Choosing a tracker or re-identification strategy, or debugging ID switches
- Trading accuracy against latency, or checking a model against a latency budget
- Deciding between more data, better labels, fine-tuning, and an architecture swap

## Key Patterns

### Set Prediction vs Dense Prediction

DETR-family models (RF-DETR, DINO, Co-DETR, Deformable DETR) emit a fixed number of queries matched one-to-one, so there is no NMS. That makes them robust in crowded scenes and gives a static-shape ONNX graph — no NMS plugin, no dynamic output, no drift between the Python reference and the C++ serving path.

YOLO-family models (YOLOX, YOLOv8+) predict densely over FPN levels and require an NMS strategy at export time. In exchange they deliver the best latency-per-accuracy on constrained hardware and are more forgiving on small datasets. YOLOX is Apache-2.0, which matters when Ultralytics' AGPL-3.0 does not fit.

### Small Objects Are a Stride Problem

```python
from loguru import logger


def recommend_small_object_strategy(
    median_object_px: float,
    frame_width: int,
    min_feature_stride: int,
) -> str:
    """Pick a small-object mitigation from the geometry, not from vibes."""
    cells = median_object_px / min_feature_stride
    logger.info("median={:.1f}px stride={} -> {:.2f} cells", median_object_px, min_feature_stride, cells)
    if cells >= 2.0:
        return "geometry is fine; investigate labels and assignment"
    if frame_width > 1920:
        return "raise input resolution before touching the architecture"
    if cells < 0.75:
        return "add a P2 head and/or tile inference"
    return "add a shallower FPN level"
```

Order of return on effort: raise input resolution, attach a head to a shallower feature level (P2/stride 4), tile inference SAHI-style with overlap larger than the biggest target, then raise query count and `max_det`, then check that label assignment produces any positives at all for sub-stride objects.

### Tracking and Identity

ByteTrack is the right first tracker; BoT-SORT when the broadcast camera pans or zooms, because camera-motion compensation removes more ID switches than any appearance model. Appearance re-ID is weak in team sports — matching kits mean the durable identity signals are jersey number voted over a track and continuity in pitch coordinates. Evaluate with HOTA and IDF1, not MOTA.

### Detector Plus Crop Model

When localization is class-agnostic and the classes are fine-grained (jersey numbers, team assignment), a single-class detector plus a second-stage crop model concentrates all localization data in one place, lets the second stage run at a higher resolution, and allows independent retraining. Jersey numbers are a sequence task for a scene-text head, not 100-way classification. Per-match clustering of DINOv2/CLIP embeddings assigns teams with essentially no labels.

## Anti-Patterns

- Do not run an architecture bake-off before the dataset is stable — you will be measuring label noise, not models
- Do not answer "which model" without object size in pixels, objects per frame, and an end-to-end latency budget
- Do not train at one resolution and serve at another, or use a different resize/color implementation in serving than in evaluation
- Do not split train/val by frame in video data — split by clip or match, or validation scores are meaningless
- Do not gate a release on mean mAP; gate on the worst slice (size bucket, venue, lighting, camera, team)
- Do not chase SOTA while labels, resolution, or a letterbox bug are the actual bottleneck
- Do not benchmark only the forward pass — decode, preprocess, NMS, and box decode frequently dominate
- Do not commit to a variant before verifying its ONNX/TensorRT export path

## Combines Well With

- **PyTorch Lightning** — training the selected architecture
- **Hydra Config** — sweeping architectures and resolutions as configuration
- **W&B** — comparing candidate architectures across runs
- **ONNX / TensorRT** — export, engine building, and end-to-end latency measurement
- **Model Evaluation** — per-class AP, slice analysis, and tracking metrics
- **FastAPI** — serving the chosen model

## Full Reference

See [`skills/cv-model-selection/SKILL.md`](https://github.com/ortizeg/whet/blob/main/skills/cv-model-selection/SKILL.md) for the full DETR-vs-YOLO comparison table, the situation-to-approach decision table, high-motion and temporal guidance, and the complete pitfalls section.
