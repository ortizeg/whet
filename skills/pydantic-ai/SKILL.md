---
name: pydantic-ai
description: >
  Use this skill whenever the project calls an LLM or VLM and needs a typed, validated
  result — structured extraction, auto-labeling images with Gemini/Moondream, JSON
  outputs, tool/function calling, or an agent loop. Reach for it any time you would
  otherwise parse a model's free-text response by hand. Builds type-safe LLM pipelines
  with PydanticAI: Agent, result_type, dependency injection, tools, retries, and
  provider-agnostic model config (Gemini, Anthropic, OpenAI, local).
---

# PydanticAI

PydanticAI brings the Pydantic "parse, don't validate" discipline to LLM/VLM calls: you
declare the result you want as a Pydantic model, and the framework enforces it — with
automatic retries when the model returns something off-schema. Use it instead of prompting
for JSON and hand-parsing the string, which is where CV/ML pipelines silently break.

The canonical use here is **VLM-in-the-loop labeling**: send an image to Gemini or a local
Moondream/DINO model and get back a *validated* set of detections or attributes, ready to
feed a training pipeline.

## The core: a typed Agent

```python
from __future__ import annotations

from loguru import logger
from pydantic import BaseModel, Field
from pydantic_ai import Agent


class Detection(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox_xyxy: tuple[float, float, float, float]


class ImageAnnotation(BaseModel):
    """The exact shape the pipeline needs back — nothing else is accepted."""

    detections: list[Detection]
    scene: str
    is_occluded: bool


annotator = Agent(
    "google-gla:gemini-2.0-flash",
    output_type=ImageAnnotation,
    system_prompt=(
        "You are a precise vision annotator. Return every visible object with a tight "
        "bounding box in pixel xyxy. Do not invent objects."
    ),
)

result = annotator.run_sync("Annotate this frame.")  # + image input, see below
logger.info("got {} detections", len(result.output.detections))
annotation: ImageAnnotation = result.output  # already validated
```

`result.output` is a validated `ImageAnnotation`. If the model returns bad JSON or a
confidence of `1.4`, PydanticAI raises and retries against the schema instead of handing
you garbage.

## Sending images (VLM inputs)

```python
from pathlib import Path

from pydantic_ai import BinaryContent, ImageUrl

# local file (e.g. a training frame)
result = annotator.run_sync([
    "Annotate this frame.",
    BinaryContent(data=Path("frame.jpg").read_bytes(), media_type="image/jpeg"),
])

# or a remote asset
result = annotator.run_sync(["Annotate this.", ImageUrl(url="https://.../frame.jpg")])
```

## Dependency injection (typed context)

Pass typed runtime dependencies (clients, thresholds, the current dataset) into the agent
and its tools — no globals.

```python
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext


@dataclass
class Deps:
    min_confidence: float
    class_whitelist: set[str]


agent = Agent("google-gla:gemini-2.0-flash", deps_type=Deps, output_type=ImageAnnotation)


@agent.system_prompt
def constrain(ctx: RunContext[Deps]) -> str:
    allowed = ", ".join(sorted(ctx.deps.class_whitelist))
    return f"Only label these classes: {allowed}. Drop boxes below {ctx.deps.min_confidence}."


result = agent.run_sync("Annotate.", deps=Deps(min_confidence=0.5, class_whitelist={"person", "ball"}))
```

## Tools (function calling)

Register Python functions the model may call; return values are typed and validated.

```python
@agent.tool
def lookup_class_id(ctx: RunContext[Deps], name: str) -> int:
    """Resolve a class name to the dataset's integer id."""
    return DATASET_CLASSES.index(name)
```

## Validation retries and self-correction

Raise `ModelRetry` from a tool or output validator to bounce a bad response back to the
model with a reason. Cap attempts so a stubborn model fails loudly.

```python
from pydantic_ai import Agent, ModelRetry

agent = Agent("google-gla:gemini-2.0-flash", output_type=ImageAnnotation, retries=2)


@agent.output_validator
def boxes_in_bounds(result: ImageAnnotation) -> ImageAnnotation:
    for d in result.detections:
        x1, y1, x2, y2 = d.bbox_xyxy
        if x2 <= x1 or y2 <= y1:
            raise ModelRetry(f"Degenerate box for {d.label}: {d.bbox_xyxy}")
    return result
```

## Provider-agnostic model config

The model is a string or an object — swap providers without touching pipeline code.

```python
# strings: "<provider>:<model>"
Agent("google-gla:gemini-2.0-flash")     # Gemini (AI Studio)
Agent("anthropic:claude-sonnet-4-5")     # Claude
Agent("openai:gpt-4o")                   # OpenAI

# a local/OpenAI-compatible endpoint (e.g. a served Moondream/vLLM)
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

local = OpenAIModel("moondream2", provider=OpenAIProvider(base_url="http://localhost:8000/v1"))
Agent(local, output_type=ImageAnnotation)
```

## Batch labeling a dataset

```python
def label_dataset(paths: list[Path], deps: Deps) -> list[ImageAnnotation]:
    out: list[ImageAnnotation] = []
    for p in paths:
        try:
            res = agent.run_sync(["Annotate.", BinaryContent(p.read_bytes(), "image/jpeg")], deps=deps)
            out.append(res.output)
        except Exception as exc:  # noqa: BLE001 — log and skip, never crash the whole run
            logger.warning("skipped {}: {}", p.name, exc)
    return out
```

## Testing (no live model calls)

Use `TestModel` / `FunctionModel` and `Agent.override` so unit tests never hit a provider.

```python
from pydantic_ai import models
from pydantic_ai.models.test import TestModel

models.ALLOW_MODEL_REQUESTS = False  # fail if any test accidentally calls a real model

def test_annotator_parses() -> None:
    with agent.override(model=TestModel()):
        result = agent.run_sync("Annotate.")
    assert isinstance(result.output, ImageAnnotation)
```

## Conventions

- **Always set `output_type`** to a Pydantic model — never parse free text.
- **Keep result models strict** (bounded floats, enums for classes) so the retry loop
  catches hallucinated values; pairs with the `pydantic` skill.
- **Inject clients/thresholds via `deps_type`**, not module globals.
- **Cap `retries`** and log-and-skip per item in batch jobs so one bad frame can't kill a
  dataset pass.
- **Gate tests** with `ALLOW_MODEL_REQUESTS = False` and `TestModel`.

## Anti-patterns

- Prompting for JSON and `json.loads`-ing the reply — no schema enforcement, silent drift.
- Putting the API client in a global — untestable, unswappable.
- Unbounded retries — a stubborn model spins cost; cap and fail.
- Trusting VLM confidences/boxes unchecked — validate ranges and geometry in an
  `output_validator`.
