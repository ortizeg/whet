# Pydantic Skill

The Pydantic Skill enforces disciplined use of Pydantic V2 across the entire codebase, operating at two distinct levels. Level 1 requires that all configuration objects -- model hyperparameters, data pipeline settings, training arguments, deployment parameters -- inherit from `BaseModel` with strict validation, frozen immutability, and explicit field definitions. Level 2 extends this discipline to internal data structures such as prediction results, evaluation metrics, and API request/response schemas. Together, these levels ensure that no unvalidated dictionary or loosely-typed dataclass sneaks into the project.

This skill covers the full surface area of Pydantic V2: `Field` definitions with constraints, custom validators using `field_validator` and `model_validator` decorators, frozen configurations that prevent accidental mutation, nested model composition, discriminated unions for polymorphic configs, and selective use of `validate_call` on functions where runtime argument validation adds genuine safety. It explicitly discourages overuse of `validate_call` on internal functions where static type checking suffices.

## When to Use

- When defining any configuration that will be loaded from YAML, JSON, or environment variables.
- When creating data transfer objects between system components (e.g., prediction results, API payloads).
- When you need runtime validation guarantees beyond what mypy provides statically.
- When building nested or polymorphic configurations that require discriminated union patterns.

## Key Features

- **Two-level adoption** -- Level 1 for configs (mandatory), Level 2 for data structures (recommended), with clear guidance on when each applies.
- **Frozen configs** -- all configuration models use `model_config = ConfigDict(frozen=True)` to prevent post-initialization mutation.
- **Field constraints** -- enforced use of `Field()` with `ge`, `le`, `pattern`, `min_length`, and other validators for every parameter.
- **Nested composition** -- patterns for composing complex configs from smaller, reusable sub-models.
- **Serialization patterns** -- consistent `model_dump()` and `model_validate()` usage for YAML/JSON round-tripping.
- **Selective validate_call** -- guidelines on when `validate_call` adds value (public APIs, CLI entry points) versus where it adds unnecessary overhead.

## Related Skills

- **[PyTorch Lightning](../pytorch-lightning/)** -- consumes Pydantic configs for all model, data, and trainer parameters.
- **[Code Quality](../code-quality/)** -- mypy strict mode complements Pydantic's runtime validation with static type safety.
- **[Abstraction Patterns](../abstraction-patterns/)** -- Protocol and ABC interfaces often carry Pydantic config objects as constructor arguments.
- **[Library Review](../library-review/)** -- evaluates third-party libraries partly on their compatibility with Pydantic-typed interfaces.
