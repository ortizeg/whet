# PydanticAI Skill

## Purpose

This skill teaches Claude how to build type-safe LLM and VLM pipelines with PydanticAI —
getting **validated, structured results** from models instead of parsing free text by
hand. The flagship use case is VLM-in-the-loop labeling: send an image to Gemini or a
local Moondream/DINO model and get back a validated set of detections or attributes ready
to feed a training pipeline.

## When to Use

- Any time the project calls an LLM or VLM and needs a typed, validated result
- Structured extraction / auto-labeling of images
- JSON outputs, tool / function calling, or an agent loop
- Replacing hand-written parsing of a model's free-text response

## Key Patterns

- `Agent` with an `output_type` Pydantic model — schema is enforced, not hoped for
- Image inputs via `BinaryContent` / `ImageUrl`
- Typed dependency injection (`deps_type`) for clients and thresholds
- `output_validator` + `ModelRetry` for self-correcting, bounded retries
- Provider-agnostic model config (Gemini, Anthropic, OpenAI, local OpenAI-compatible)
- `TestModel` + `ALLOW_MODEL_REQUESTS = False` for hermetic tests

Pairs with `pydantic` (strict result models) and `model-evaluation` (scoring the
labels the VLM produces).
