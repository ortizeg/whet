# Mutable Data Models

Non-frozen Pydantic models for state that legitimately changes during processing: training metrics, detection results, per-frame aggregates.

## Contents

- [When to Drop `frozen`](#when-to-drop-frozen)
- [Rules](#rules)
- [Notes](#notes)

## When to Drop `frozen`

Use non-frozen models for data structures that need to be updated during processing, such as
tracking state, results, or intermediate computations. Configuration stays frozen; *data* that
accumulates does not.

```python
"""Mutable data structures for tracking training progress."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class TrainingMetrics(BaseModel):
    """Mutable metrics tracked during training."""

    model_config = {"extra": "forbid"}

    epoch: int = 0
    train_loss: float = 0.0
    val_loss: float = float("inf")
    best_val_loss: float = float("inf")
    learning_rate: float = 0.0
    samples_processed: int = 0

    def update_epoch(self, train_loss: float, val_loss: float, lr: float) -> None:
        """Update metrics after completing an epoch."""
        self.epoch += 1
        self.train_loss = train_loss
        self.val_loss = val_loss
        self.learning_rate = lr
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss


class DetectionResult(BaseModel):
    """Single object detection result. bbox is (x1, y1, x2, y2)."""

    model_config = {"extra": "forbid"}

    class_id: int = Field(ge=0)
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: tuple[float, float, float, float]

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, v: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        """Ensure bounding box coordinates are valid."""
        x1, y1, x2, y2 = v
        if x2 <= x1 or y2 <= y1:
            msg = f"Invalid bbox: x2 must be > x1 and y2 must be > y1, got ({x1}, {y1}, {x2}, {y2})"
            raise ValueError(msg)
        return v


class FrameDetections(BaseModel):
    """All detections in a single frame."""

    model_config = {"extra": "forbid"}

    frame_id: int = Field(ge=0)
    timestamp: float = Field(ge=0.0)
    detections: list[DetectionResult] = Field(default_factory=list)
    processing_time_ms: float = Field(ge=0.0)

    @property
    def num_detections(self) -> int:
        """Return total number of detections."""
        return len(self.detections)

    def filter_by_confidence(self, threshold: float) -> list[DetectionResult]:
        """Return detections above the confidence threshold."""
        return [d for d in self.detections if d.confidence >= threshold]
```

## Rules

1. **Omit `frozen = True`** -- these models need to be mutable.
2. **Still set `extra = "forbid"`** -- even mutable models should reject unknown fields.
3. **Add methods for common operations** -- like `update_epoch()` and `filter_by_confidence()`.
4. **Use properties for derived values** -- like `num_detections`.

## Notes

- **Field constraints still apply on assignment only if you opt in.** By default Pydantic V2
  validates on construction, not on attribute assignment. Add `"validate_assignment": True` to
  `model_config` if mutations must stay within the declared bounds — worth it for models like
  `TrainingMetrics` whose fields are written throughout a run.
- **`Field(default_factory=list)` for the `detections` list.** A bare `default=[]` would share
  one list across every instance — the classic mutable-default bug.
- **`float("inf")` as the initial `val_loss`/`best_val_loss`** makes the first epoch's
  comparison unconditionally true, so no special-casing is needed in `update_epoch`.
- **Domain methods belong on the model.** `update_epoch` and `filter_by_confidence` keep the
  invariants (best-loss tracking, threshold semantics) in one place instead of duplicated at
  every call site.
- **Properties are not fields.** `num_detections` is computed on access and never serialized by
  `model_dump()`; if the value must appear in the output, use `@computed_field`.
- The bbox validator enforces a real invariant that no `Field` constraint can express: ordering
  between two elements of the same tuple.
