---
name: abstraction-patterns
description: >
  Use this skill when deciding whether and how to introduce an abstraction, wrapper,
  base class, or interface in an AI/CV Python codebase — refactoring repeated logic,
  wrapping a third-party API, or judging whether code is over-engineered. Applies the
  Rule of Three and interface-design principles to cut cognitive load without premature
  generalization. Reach for it any time you'd otherwise copy-paste a third variation of
  the same logic or spin up a class for a single function, even if the user doesn't say
  "abstraction". Not for choosing which library to depend on (see library-review) or for
  type/data validation (see pydantic).
---

# Abstraction Patterns Skill

Well-abstracted Python for AI/CV projects. Abstraction exists to reduce cognitive
load, not to add layers — every abstraction must justify itself by making at least
three call sites simpler. This page carries the decision rule and the two shapes
that cover most cases; the deep dives hold the full worked patterns.

## The Rule: When to Abstract

Abstract when:
- **Three or more call sites** share the same logic
- **Resource management** requires setup/teardown (video readers, model sessions, database connections)
- **Complex validation** needs to happen consistently (image format checking, bounding box validation)
- **External dependencies** need to be isolated for testing (file I/O, API calls, hardware access)

Do NOT abstract when:
- There is only one call site (inline it)
- The abstraction hides important details (GPU memory management, batch dimension handling)
- A simple function would suffice (do not create a class for a single method)

## Shape 1: A function, until proven otherwise

The default abstraction is a module-level function with a precise signature and
explicit errors. Reach for a class only when there is state to hold or resources to
manage.

```python
def load_image(path: str | Path, color_space: ColorSpace = "rgb") -> np.ndarray:
    """Load an image with validation. Raises FileNotFoundError / ValueError / RuntimeError."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)  # BGR
    if image is None:
        raise RuntimeError(f"Failed to decode image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if color_space == "rgb" else image
```

If you find yourself writing `class XProcessor` whose only job is to store one
parameter and expose one method, use `functools.partial` instead.

## Shape 2: A context manager for anything that must be released

File handles, capture devices, model sessions, and connections all leak when the
release path is left to the caller. Own it in `__enter__`/`__exit__`:

```python
class VideoReader:
    """Context-managed OpenCV video reader."""

    def __enter__(self) -> VideoReader:
        self._cap = cv2.VideoCapture(str(self._path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self._path}")
        return self

    def __exit__(self, *args: object) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            yield frame


with VideoReader("input.mp4") as reader:
    for frame in reader:
        visualize(frame, model.predict(frame))
```

Methods that require the managed resource should raise a clear `RuntimeError` when
used outside the `with` block rather than failing on a `None` attribute.

## Shape 3: A frozen dataclass for the values that travel together

When several call sites pass around the same tuple of values, name it. A frozen
dataclass documents the shape, prevents accidental mutation, and gives derived
values a home — without introducing behaviour the caller has to learn.

```python
@dataclass(frozen=True)
class VideoMetadata:
    """Immutable metadata for a video file."""

    path: Path
    fps: float
    total_frames: int
    width: int
    height: int

    @property
    def duration_seconds(self) -> float:
        return self.total_frames / self.fps if self.fps > 0 else 0.0
```

This is the cheapest abstraction available: no inheritance, no lifecycle, no hidden
control flow. Prefer it before reaching for a class with methods.

## Conventions

- **Count call sites before abstracting** — three real uses, not three imagined ones.
- **Type every boundary** — the whole point of a wrapper is that call sites stop guessing shapes and dtypes.
- **Raise specific exceptions** with the offending value in the message; never return `None` for failure.
- **Keep configuration explicit** — frozen dataclasses or Pydantic models over loose kwargs dicts.
- **One interface per concept** — if several implementations must be interchangeable, define the ABC once (`update`/`compute`/`reset`) and let the call site stay ignorant of which one it holds.
- **Make the abstraction testable without hardware** — if a wrapper can only be tested with a real camera or GPU, the boundary is wrong.
- **Prefer composition over inheritance** — a collection that drives several implementations beats a deep class hierarchy.

## Anti-patterns

- **Abstracting single-use logic** — one call site means inline it; a class wrapping a single `cv2.resize` is noise.
- **Hiding critical details** — an "auto batcher" that silently picks batch size from GPU memory removes the developer's ability to reason about OOM.
- **A class for a single function** — `NMSProcessor(threshold).process(...)` is just `partial(nms, iou_threshold=...)`.
- **Premature base classes** — an ABC with exactly one subclass is indirection with no payoff.
- **Leaky resource wrappers** — exposing the raw `cv2.VideoCapture` alongside the wrapper invites callers to bypass cleanup.

## Deep dives

- `references/wrapper-patterns.md` — read when writing a full wrapper around a third-party API: complete `VideoReader`, validated image load/save, and an inference wrapper with preprocessing and batching.
- `references/interface-design.md` — read when several implementations must be interchangeable and you need an ABC plus a collection that drives them (metrics example).
- `references/when-not-to-abstract.md` — read when judging whether existing code is over-engineered, or before adding a class you are not sure earns its place.
- `references/testing-abstractions.md` — read when writing tests that pin an abstraction's contract: accumulation, reset semantics, and error paths.
