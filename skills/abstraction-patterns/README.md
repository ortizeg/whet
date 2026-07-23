# Abstraction Patterns Skill

The Abstraction Patterns Skill defines design patterns for wrapping external libraries so that third-party APIs never leak into business logic. CV/ML projects are especially vulnerable to tight coupling: training code calls PyTorch directly, data pipelines embed OpenCV function signatures, and experiment tracking is wired to a specific vendor's SDK. When a library introduces breaking changes, gets abandoned, or needs to be swapped, the cost of untangling these dependencies is enormous. This skill prevents that by establishing clear abstraction boundaries through ABC classes, Protocol interfaces, registry patterns, and strategy patterns.

The core principle is simple: business logic depends on interfaces you own, never on third-party implementations directly. A `ModelTrainer` protocol defines what training looks like; a `LightningTrainer` adapter implements it using PyTorch Lightning. An `ImageReader` ABC defines how images are loaded; an `OpenCVImageReader` provides the concrete implementation. Registry patterns allow runtime selection of implementations by name, and strategy patterns enable swapping algorithms without modifying calling code. Crucially, this skill also defines when not to abstract -- thin wrappers around stable, ubiquitous libraries like `pathlib` or `logging` add indirection without value.

## When to Use

- When integrating any third-party library into the project for the first time.
- When business logic needs to be testable in isolation without external dependencies.
- When you anticipate needing to swap between alternative implementations (e.g., OpenCV vs. Pillow, MLflow vs. W&B).
- When designing plugin-style architectures that allow runtime registration of new implementations.

## Key Features

- **ABC interfaces** -- abstract base classes with `@abstractmethod` for heavyweight contracts requiring inheritance and shared state.
- **Protocol interfaces** -- structural typing via `Protocol` for lightweight contracts where duck typing is preferred over inheritance.
- **Registry pattern** -- dictionary-based or decorator-based registration of implementations, enabling runtime lookup by name or config string.
- **Strategy pattern** -- interchangeable algorithm implementations selected at construction time through dependency injection.
- **Anti-patterns guide** -- explicit guidance on when abstraction adds unnecessary complexity (stable APIs, single-implementation cases, prototyping).
- **Testing patterns** -- mock and fake implementations of interfaces for unit testing business logic without external dependencies.

## Related Skills

- **[Library Review](../library-review/)** -- evaluates libraries before adoption, determining whether and how they should be wrapped.
- **[Pydantic](../pydantic/)** -- configuration objects are passed to abstract interfaces, decoupling parameter definitions from implementations.
- **[PyTorch Lightning](../pytorch-lightning/)** -- Lightning modules implement task-specific interfaces defined through abstraction patterns.
- **[Code Quality](../code-quality/)** -- mypy strict mode verifies that implementations correctly satisfy Protocol and ABC contracts.
