# Expert Coder Agent

The Expert Coder Agent is your primary coding assistant for all development tasks in AI/CV projects.

## Purpose

This agent guides all code generation to follow project standards:
- Abstraction-first design
- Pydantic for all configs and data structures
- Full type safety with mypy strict mode
- Testable, maintainable code

## Strictness Level

**ADVISORY** — This agent guides but doesn't block.

## When to Use

- Creating new modules or classes
- Refactoring existing code
- Implementing new features
- Code reviews during development

## Example Session

```
You: "I need a video processing pipeline that applies object detection frame by frame"

Expert Coder: "I'll create a pipeline with proper abstractions:
1. VideoReader abstraction (wraps cv2)
2. Detector interface (abstract base class)
3. Pipeline orchestrator (Pydantic config)
4. Full type hints and docstrings"
```

## Related Skills

- `pydantic` — Config and data structure patterns
- `abstraction-patterns` — When and how to abstract
- `code-quality` — Type hints and formatting rules
- `pytorch-lightning` — ML-specific patterns
