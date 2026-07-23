# PydanticAI

The pydantic-ai skill covers building type-safe LLM and VLM pipelines with PydanticAI — getting validated, structured results from models instead of parsing free text by hand.

**Skill directory:** `skills/pydantic-ai/`

## Purpose

PydanticAI brings the "parse, don't validate" discipline to LLM/VLM calls: declare the result you want as a Pydantic model and the framework enforces it, retrying automatically when the model returns something off-schema. This skill teaches Claude Code to use it for the flagship CV use case — VLM-in-the-loop labeling, where an image goes to Gemini or a local Moondream/DINO model and comes back as a validated set of detections or attributes ready to feed a training pipeline.

## When to Use

- Any time the project calls an LLM or VLM and needs a typed, validated result
- Structured extraction / auto-labeling of images
- JSON outputs, tool / function calling, or an agent loop
- Replacing hand-written parsing of a model's free-text response

## Key Patterns

- `Agent` with an `output_type` Pydantic model
- Image inputs via `BinaryContent` / `ImageUrl`
- Typed dependency injection (`deps_type`)
- `output_validator` + `ModelRetry` for self-correcting, bounded retries
- Provider-agnostic model config (Gemini, Anthropic, OpenAI, local)
- `TestModel` + `ALLOW_MODEL_REQUESTS = False` for hermetic tests

## Anti-Patterns

- Prompting for JSON and `json.loads`-ing the reply — no schema enforcement, silent drift
- Putting the API client in a global — untestable, unswappable
- Unbounded retries — cap and fail loudly
- Trusting VLM confidences/boxes unchecked — validate ranges and geometry
