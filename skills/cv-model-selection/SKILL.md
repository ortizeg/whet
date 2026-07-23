---
name: cv-model-selection
description: >
  Use this skill when choosing or changing a computer-vision model architecture — picking
  a detector (RF-DETR / DINO / Co-DETR vs YOLOX / YOLOv8+), deciding between one detector
  and a detector-plus-crop-classifier, choosing a tracker or re-ID approach, trading
  accuracy against latency, or diagnosing a small-object / high-motion regime. Reach for
  it any time the question is "which model should I use", "why is mAP_small terrible",
  "should I fine-tune or switch architectures", "do I need tracking", or "will this hit my
  latency budget" — even if the user doesn't say the word "architecture" and only describes
  a symptom. Not for training-loop mechanics (see pytorch-lightning), config plumbing (see
  hydra-config), run tracking (see wandb), export and engine building (see onnx, tensorrt),
  metric computation (see model-evaluation), or dataset splitting (see data-pipelines).
---

# CV Model Selection

Architecture choice is a *regime* decision, not a leaderboard decision. Two projects with
identical COCO numbers behave completely differently when the objects are 8 pixels wide,
the camera pans, and the budget is 12 ms per frame. This skill covers the tradeoffs that
actually decide a CV project.

Prerequisite framing: **quantify the regime before naming a model.** Object size
distribution in pixels, objects per frame, frames per second required, latency budget
end-to-end, labeled image count, and label noise level. Those six numbers determine the
answer more than any benchmark table.

## Detection: DETR-family vs YOLO-family

The real axis is **set prediction vs dense prediction**, not "transformer vs CNN".

| Dimension | DETR-family (RF-DETR, DINO, Deformable DETR, Co-DETR) | YOLO-family (YOLOX, YOLOv8+) |
|---|---|---|
| Output | Fixed `num_queries` predictions, one-to-one Hungarian matching | Dense per-location predictions over FPN levels |
| Post-process | None — NMS-free | NMS (or NMS-free variants with extra heads) |
| Assignment | Bipartite matching (+ denoising queries in DINO/RF-DETR) | SimOTA (YOLOX) / TaskAligned (v8+) |
| Convergence | Historically slow; modern variants train in tens of epochs | Fast, well-trodden recipes |
| Crowding | Capped by query count; one-to-one matching handles overlap well | NMS merges/deletes overlapping true positives |
| Export | Static output shape — clean ONNX; deformable attention may need care | Needs NMS strategy (external, ONNX op, or TRT plugin) |
| Small data | More sensitive; benefits from strong pretrained backbones | Robust, heavy augmentation recipes carry it |
| Licensing | RF-DETR Apache-2.0; DINO/Co-DETR mostly permissive research code | YOLOX Apache-2.0; Ultralytics is AGPL-3.0 |

### Choose DETR-family when

- **Crowded/overlapping objects.** NMS is the failure mode in a ruck, a scrum, a rebound
  box-out. One-to-one set prediction does not delete a true positive because it overlaps
  another true positive.
- **The deploy path is ONNX/TensorRT and you want it boring.** No NMS means a fixed
  `[B, num_queries, 4]` + `[B, num_queries, C]` output. No dynamic shapes, no plugin, no
  divergence between the Python reference and the C++ serving path. This is the single
  most underrated argument for RF-DETR in production.
- **You want one knob for the accuracy/latency curve.** RF-DETR-style models scale by
  input resolution and backbone size; you can re-tune the operating point without changing
  the export or serving code.
- **Maximum accuracy, latency irrelevant** (offline analytics, auto-labeling, pseudo-label
  generation): Co-DETR / DINO with a large backbone. Treat these as *label factories*, not
  as deployment targets.

Watch out: deformable / multi-scale attention lowers to `grid_sample`-like ops. Verify
ONNX export and the TensorRT path **before** committing to a variant, not after training.

### Choose YOLO-family when

- **Hard real-time on modest hardware.** Edge devices, multi-stream ingest, or anything
  where you are counting milliseconds per stream.
- **Small labeled dataset and you need a result this week.** The augmentation recipe
  (mosaic, mixup, random perspective, HSV) plus dense supervision is genuinely more
  forgiving than one-to-one matching on a few thousand images.
- **Very high object counts per frame** beyond a comfortable query budget.
- **YOLOX specifically** when licensing matters — anchor-free, decoupled head, SimOTA,
  Apache-2.0, and the export story is well documented. Prefer it over AGPL alternatives
  for anything shipping commercially.

### The honest default for sports CV

Start RF-DETR for the primary player/ball detector (crowding + clean export), keep YOLOX
as the latency escape hatch and as a fast baseline to sanity-check labels. Do **not** run
an architecture bake-off before the dataset is stable — you will be measuring label noise.

```python
from enum import Enum

from loguru import logger
from pydantic import BaseModel, Field, model_validator


class DetectorFamily(str, Enum):
    RF_DETR = "rf-detr"
    YOLOX = "yolox"


class DetectorConfig(BaseModel):
    """Deployment-aware detector selection."""

    model_config = {"frozen": True}

    family: DetectorFamily
    input_size: tuple[int, int] = (960, 960)
    num_queries: int = Field(default=300, ge=1)
    max_det: int = Field(default=300, ge=1)
    latency_budget_ms: float = Field(default=25.0, gt=0)
    tile_inference: bool = False

    @model_validator(mode="after")
    def warn_on_query_ceiling(self) -> "DetectorConfig":
        if self.family is DetectorFamily.RF_DETR and self.tile_inference:
            logger.warning(
                "Tiled inference multiplies effective objects per forward pass; "
                "num_queries={} may truncate crowded tiles.",
                self.num_queries,
            )
        return self
```

## Small-object regimes

This is the dominant failure mode in sports CV. A ball at 8-14 px, a distant player at
20 px, a jersey number at 12 px tall. `mAP_small` collapsing is almost never fixed by a
bigger model.

**Effective stride is the whole game.** If the deepest useful feature level has stride 32
and the object is 10 px, the object occupies less than a third of one cell. No head can
regress what the backbone has already averaged away.

Fix, in order of return on effort:

1. **Raise input resolution.** Going 640 → 1280 on a 4K broadcast frame usually moves
   small-object recall more than any architecture change. Cost is roughly quadratic in
   latency; measure before and after.
2. **Use shallower feature levels.** Add a P2 (stride 4) level to the FPN, or ensure the
   detector actually attaches a head to it. Many stock configs start at P3 (stride 8).
3. **Tile / slice inference (SAHI-style)** when resolution alone is not enough. Slice the
   frame into overlapping crops, infer per tile, map boxes back, and merge. Overlap must
   exceed the largest target object so no object is cut by every tile boundary. Cost is
   linear in tile count — often 4-9× latency, which is why it lives in offline pipelines
   or on a second-stage ROI around a coarse prediction.
4. **Raise detection capacity.** DETR-family: `num_queries` and the eval `max_det`.
   YOLO-family: NMS `max_det` and `topk` before NMS. Defaults of 100 or 300 silently
   truncate a tiled crowded frame.
5. **Check label assignment.** Center-sampling radii and SimOTA/TAL candidate selection
   can produce *zero* positive samples for an object smaller than the stride. If a class
   never gets positives, it never learns — this looks identical to "the model is too small".

Two evaluation traps: COCO defines "small" as area < 32² = 1024 px², which lumps a
14 px ball together with a 30 px helmet. Define your own area buckets (e.g. <12², 12²-24²,
24²-48², >48²) and report recall per bucket. And never resize a 4K frame down to 640 in
the eval transform while serving at 1280 — see the pitfalls section.

```python
from loguru import logger


def recommend_small_object_strategy(
    median_object_px: float,
    frame_width: int,
    min_feature_stride: int,
) -> str:
    """Pick a small-object mitigation from the geometry, not from vibes."""
    cells = median_object_px / min_feature_stride
    logger.info(
        "median_object={:.1f}px stride={} -> {:.2f} feature cells",
        median_object_px,
        min_feature_stride,
        cells,
    )
    if cells >= 2.0:
        return "geometry is fine; investigate labels and assignment"
    if frame_width > 1920:
        return "raise input resolution before touching the architecture"
    if cells < 0.75:
        return "add a P2 head and/or tile inference"
    return "add a shallower FPN level"
```

## High-motion and temporal regimes

- **Motion blur is a labeling problem first.** A blurred ball is an ellipse or a streak.
  Decide the annotation convention (tight box on the streak vs. estimated instantaneous
  position) and enforce it, or the model learns a bimodal target. Directional-blur
  augmentation only helps once the labels are consistent.
- **Broadcast artifacts**: rolling shutter skew, interlacing, transcoding blocking, and
  hard cuts. Train on frames decoded by the *same* decoder settings used in production.
- **Frame sampling.** Consecutive frames are near-duplicates. Sample sparsely for training
  (temporal stride, or cluster-and-sample by embedding), and split train/val by **clip or
  match**, never by frame — see `data-pipelines`. Frame-level splits inflate validation
  scores enormously and are the most common silent bug in video CV.
- **Per-frame detection is insufficient when** the answer depends on time: the ball is
  fully occluded for 15 frames, the event is "pass" vs "shot", possession must be
  attributed, or the output must be temporally smooth for a downstream consumer.

Escalation ladder, cheapest first:

1. Per-frame detector + tracker with a motion model (fills short gaps, smooths jitter).
2. Test-time temporal aggregation: run the detector on frame *t* but bias/seed it with the
   previous frame's boxes (ROI crops, propagated queries).
3. Multi-frame input: stack N frames or add a lightweight temporal module. Real accuracy
   gain on occlusion, real cost in latency, memory, and export complexity.
4. A dedicated temporal model (action recognition / event spotting) as a *second* stage
   consuming tracks, not raw pixels. Usually the best accuracy-per-engineering-hour.

## Tracking and re-ID

Detection alone is rarely the deliverable in sports — the deliverable is a per-identity
trajectory. Trackers are cheap; pick by association signal.

| Approach | Association signal | Use when |
|---|---|---|
| SORT | Kalman + IoU | Baseline only |
| ByteTrack | Kalman + IoU, **including low-score detections** | Strong detector, moderate occlusion. Best first choice. |
| OC-SORT | Observation-centric motion recovery | Non-linear motion, frequent short occlusions |
| BoT-SORT | ByteTrack + camera-motion compensation + appearance | **Panning/zooming broadcast cameras** |

Key points that actually change outcomes:

- **Camera motion compensation matters more than appearance embeddings** in broadcast
  footage. A pan of a few pixels per frame breaks IoU association for small fast objects
  long before appearance would have helped.
- **Appearance re-ID is weak in team sports.** Players on the same team wear the same kit;
  a generic person re-ID embedding gives you "which team" and little else. The durable
  identity signals are jersey number (OCR/STR over time, voted across frames), pitch
  position continuity, and pose/gait — not a ReID cosine distance.
- **Track in world coordinates when you can.** Homography to pitch coordinates turns
  camera-induced motion into zero motion and makes a constant-velocity model correct.
  This single change removes more ID switches than any tracker swap.
- **Measure with HOTA and IDF1**, not MOTA. MOTA is dominated by detection recall and will
  happily reward a tracker that shreds identities. See `model-evaluation`.
- **Ball tracking is a different problem from player tracking.** One instance, tiny, high
  velocity, frequently occluded, and no appearance signal. Use a dedicated small-object
  detector plus a physics-aware filter or trajectory fit; do not expect a generic
  multi-object tracker to handle it.

## Classification and embedding backbones

**Prefer detector + crop-classifier over a many-class detector when localization is
class-agnostic and the classes are fine-grained.** Jersey-number reading is the canonical
case:

- A 100-class "number" detector splits an already-small dataset across 100 heads and still
  has to localize. A single-class person/torso detector concentrates all localization data,
  and a second-stage model reads the crop at whatever resolution it needs (often *higher*
  than the detector's input).
- Jersey number is a **sequence** task, not 100-way classification. Use a scene-text
  recognition head (CTC or attention decoder) so "7", "07", and "77" share structure.
- The two stages can be retrained independently. Relabeling numbers does not force a
  detector retrain — a large practical win.
- Cost: two forward passes and a crop pipeline; error compounds through the cascade.

Use a single multi-class detector when classes are visually separable at detector
resolution and roughly balanced (player / referee / ball / goalpost).

**Embedding models** (DINOv2, CLIP, or a fine-tuned metric-learning backbone) earn their
place for:

- **Team assignment by clustering** crop embeddings *within a match*. Kits change every
  game; a trained team classifier does not generalize, but per-match k-means on embeddings
  does, with essentially no labels.
- **Near-duplicate removal and active-learning mining** — embed the unlabeled pool, find
  what is far from the labeled set, label that. Usually a bigger accuracy win than an
  architecture change.
- **Few-shot retrieval** (find every frame containing this logo/equipment) without training
  a detector.

## Decision table

| Situation | Recommended approach | Why |
|---|---|---|
| Players, crowded, real-time, GPU serving | RF-DETR at the largest resolution inside budget | NMS-free handles overlap; static export to ONNX/TensorRT |
| Same, but edge device or many streams per GPU | YOLOX + TensorRT, tuned NMS `max_det` | Best latency-per-accuracy; Apache-2.0 |
| Ball at <15 px in 4K broadcast | Dedicated small-object model: high res + P2 head, optionally ROI-tiled | Stride, not capacity, is the binding constraint |
| Offline auto-labeling / pseudo-labels | Co-DETR or DINO with a large backbone, TTA, no latency limit | Accuracy is the only objective; humans review the output |
| Need per-player trajectories | Detector + ByteTrack; BoT-SORT if the camera pans | Motion association carries; CMC handles the pan |
| Persistent identity across occlusions | Jersey-number STR voted over the track + pitch-coordinate continuity | Appearance embeddings are near-useless in matching kits |
| Jersey numbers, team assignment, fine-grained attributes | Single-class detector + crop model (STR / classifier / embedding cluster) | Concentrates data; independent retraining; higher crop resolution |
| Event/action output ("pass", "shot") | Temporal model over tracks, not raw frames | Per-frame detection has no access to the evidence |
| mAP good, production bad | Slice analysis before any model change | The gap is a distribution slice, not the architecture |
| <2k labeled images | Fine-tune a strong pretrained detector; spend the effort on labels | Architecture differences are inside the noise band at this size |

## Common pitfalls

**Class imbalance masquerading as a model problem.** One ball and twenty-two players per
frame means the ball contributes ~4% of the positives and its AP barely moves the mAP.
Report per-class AP always; consider class-balanced sampling, loss weighting, or simply a
separate ball model with its own resolution and its own success criterion.

**Train/inference resolution and preprocessing mismatch.** Training with random-resize to
640 and serving at 1280, or evaluating with torchvision resize while serving with an
OpenCV `INTER_LINEAR` on BGR, produces a real accuracy loss that no amount of retraining
explains. Pin one preprocessing implementation, and assert numerical equivalence between
the training transform and the served ONNX preprocessing in a test.

**Letterbox and aspect-ratio handling.** Pad color, stride alignment, and the
un-letterboxing math on the way out are a classic source of silently shifted boxes — the
model looks "slightly wrong everywhere" and mAP degrades a few points. Round-trip a known
box through preprocess and postprocess in a unit test. If the source aspect ratio is fixed
(broadcast 16:9), consider training at that aspect ratio and skipping letterbox entirely.

**Label quality dominates architecture choice.** Inconsistent handling of occluded balls,
missing annotations treated as background (actively teaching the model to suppress correct
detections), and disagreeing annotators cost far more than the gap between two modern
detectors. Before any architecture experiment, measure label noise: re-annotate 200 images
blind and compute agreement. If agreement is below your target accuracy, the architecture
experiment cannot be measured.

**Over-indexing on a single mAP number.** mAP averages away exactly the failures that
matter: night games, one camera angle, one team's kit, the far touchline, the smallest
size bucket. Always slice — by object size, camera, venue, lighting, and team — and gate
releases on the worst slice, not the mean. See `model-evaluation`.

**Chasing SOTA when data is the bottleneck.** A +0.5 COCO mAP architecture will not close
a 20-point gap on your worst slice. Rank candidate work by expected gain: label 2,000
targeted hard frames, fix the letterbox bug, raise resolution, then — last — swap the
architecture.

**Benchmarking the forward pass instead of the pipeline.** Decode, color convert, resize,
normalize, H2D copy, forward, NMS, D2H, and box decode all count. Preprocessing and NMS
frequently dominate a well-optimized detector. Measure end-to-end p99 under production
batch size and concurrency; see `tensorrt` for engine-level profiling and `onnx` for the
export path.

**Ignoring license and provenance until deployment.** Model weights and training code can
carry different licenses (AGPL-3.0 in particular). Decide this at selection time — not
after a month of training runs.

## Setup

```bash
pixi add pytorch torchvision
pixi add --pypi supervision   # box/tracking utilities, dataset conversion
pixi add --pypi sahi          # tiled/sliced inference for small objects
```

## Related skills

- `pytorch-lightning` — training loop, callbacks, distributed training
- `hydra-config` — sweeping architectures and resolutions as config
- `wandb` — comparing candidate architectures across runs
- `onnx` / `tensorrt` — export, engine building, precision, benchmarking
- `model-evaluation` — mAP, per-class and slice analysis, tracking metrics
- `data-pipelines` — splitting by clip/match, leakage, augmentation design
- `fastapi` — serving the chosen model
