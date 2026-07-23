# CV Model Selection Skill

## Purpose

The purpose of this skill is to guide computer-vision architecture decisions — which
detector to use, whether to add tracking, whether to split a task into a detector plus a
crop model, and how to trade accuracy against latency. It frames selection as a *regime*
decision (object size, crowding, motion, latency budget, dataset size, label quality)
rather than a leaderboard decision, and it is written for production sports CV stacks
built on RF-DETR / YOLOX with ONNX and TensorRT export.

It deliberately does not re-teach training loops, config management, experiment tracking,
export mechanics, or metric computation — those live in `pytorch-lightning`,
`hydra-config`, `wandb`, `onnx` / `tensorrt`, and `model-evaluation`.

## When to Use

Use this skill when:

- Picking a detection architecture, or deciding whether to change the one in use.
- Someone asks "which model should I use" or "is RF-DETR or YOLOX better here".
- `mAP_small` is collapsing, or small/fast objects (balls, distant players) are missed.
- Deciding whether per-frame detection is enough, or tracking / temporal modeling is needed.
- Choosing a tracker or a re-identification strategy, or debugging ID switches.
- Trading accuracy against latency, or checking whether a model fits a latency budget.
- Deciding between fine-tuning, more data, and an architecture swap.
- Deciding whether a fine-grained task (jersey numbers, team assignment) should be a
  second-stage crop model rather than extra detector classes.

## Key Patterns

- **Quantify the regime first** — object size in pixels, objects per frame, FPS, latency
  budget, labeled image count, label noise. Those decide the architecture.
- **Set prediction vs dense prediction** — DETR-family is NMS-free (clean export, crowd
  robust); YOLO-family is dense (fastest, most forgiving on small datasets).
- **Effective stride, not model capacity**, governs small-object performance. Raise
  resolution, add a P2 head, then tile (SAHI-style), then raise query / `max_det` limits.
- **Split by clip or match, never by frame** — consecutive video frames are near-duplicates.
- **Camera-motion compensation beats appearance re-ID** in broadcast footage; matching
  kits make ReID embeddings weak, so jersey number and pitch position carry identity.
- **Detector + crop model** when localization is class-agnostic and classes are
  fine-grained; embedding models (DINOv2/CLIP) for per-match team clustering and mining.
- **Slice before you swap** — a single mAP number hides the failure that is actually
  blocking production.

See `SKILL.md` for the full decision table, code examples, and the pitfalls section.
