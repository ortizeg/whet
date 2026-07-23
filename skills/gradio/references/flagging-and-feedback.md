# Flagging and Feedback Collection

Scope: subclassing `gr.FlaggingCallback` to persist structured user feedback for active learning, and wiring it into a demo.

Subclass `gr.FlaggingCallback` to persist structured feedback for active learning.

```python
"""Custom flagging callback for ML feedback collection."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
from loguru import logger
from pydantic import BaseModel


class FlaggedSample(BaseModel, frozen=True):
    timestamp: str
    flag_reason: str
    input_hash: str
    prediction: str
    user_correction: str | None = None


class MLFlaggingCallback(gr.FlaggingCallback):
    def __init__(self, output_dir: str = "flagged_data") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def setup(self, components: list, flagging_dir: str | Path) -> None:
        self.flagging_dir = Path(flagging_dir)
        self.flagging_dir.mkdir(parents=True, exist_ok=True)

    def flag(self, flag_data: list, flag_option: str = "incorrect", username: str | None = None) -> int:
        sample = FlaggedSample(
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            flag_reason=flag_option,
            input_hash=str(hash(str(flag_data[0]))),
            prediction=str(flag_data[1]) if len(flag_data) > 1 else "",
            user_correction=str(flag_data[2]) if len(flag_data) > 2 else None,
        )
        (self.output_dir / f"flag_{sample.timestamp}.json").write_text(sample.model_dump_json(indent=2))
        logger.info("Flagged sample saved (reason: {})", flag_option)
        return 1


demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(),
    flagging_callback=MLFlaggingCallback(output_dir="flagged_data"),
    flagging_options=["incorrect", "low_confidence", "interesting"],
)
```

`flag_data` arrives as a positional list in the same order as the demo's components
(inputs first, then outputs), which is why the callback indexes defensively. Validate
the record with a frozen Pydantic model before writing so malformed feedback fails at
collection time rather than during retraining.
