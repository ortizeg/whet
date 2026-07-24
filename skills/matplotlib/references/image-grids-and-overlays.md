# Image Grids and Detection Overlays

Displaying batches of images in a grid, and drawing bounding boxes with labels and scores over an image.

## Contents

- [Image Grid Display](#image-grid-display)
- [Bounding Box Overlay](#bounding-box-overlay)

## Image Grid Display

Works for augmentation previews, sample batches, etc. Grayscale images (2D) are
detected automatically.

```python
def plot_image_grid(
    images: list[np.ndarray],
    titles: list[str] | None = None,
    ncols: int = 4,
    figsize_per_image: tuple[float, float] = (3, 3),
    save_path: str | Path | None = None,
    suptitle: str | None = None,
) -> plt.Figure:
    """Display images (H,W,3 or H,W) in a grid. Returns the figure."""
    n = len(images)
    nrows = (n + ncols - 1) // ncols
    figsize = (figsize_per_image[0] * ncols, figsize_per_image[1] * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes)

    for idx in range(nrows * ncols):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        if idx < n:
            cmap = "gray" if images[idx].ndim == 2 else None
            ax.imshow(images[idx], cmap=cmap)
            if titles:
                ax.set_title(titles[idx], fontsize=10)
        ax.axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=16, y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig
```

Notes:

- **`np.atleast_2d(axes)`** normalizes the axes array so indexing works whether `plt.subplots`
  returned a scalar, a 1-D row, or a 2-D grid. Without it, single-row grids crash on `axes[row, col]`.
- **The loop runs over the full grid**, not just `len(images)`, so leftover cells still get
  `ax.axis("off")` and do not render empty framed boxes.
- **Grayscale detection** via `images[idx].ndim == 2` — passing `cmap=None` for an RGB array is
  correct, and passing `cmap="gray"` for a 2-D array avoids matplotlib's default viridis.
- Images must be in RGB order (convert from OpenCV's BGR first) and either `uint8` in [0, 255]
  or float in [0, 1]; anything else triggers clipping warnings.
- Scale `figsize_per_image` rather than the total figure size, so grids stay legible as the
  batch grows.

## Bounding Box Overlay

Matplotlib overlays are better than OpenCV for notebooks and papers (crisp text,
vector output).

```python
import matplotlib.patches as patches


def plot_detections(
    image: np.ndarray,
    boxes: np.ndarray,             # (N, 4) xyxy
    labels: list[str],
    scores: np.ndarray | None = None,
    class_colors: dict[str, str] | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (12, 8),
) -> plt.Figure:
    """Draw detection boxes with labels/scores over an RGB image. Returns the figure."""
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image)
    default_colors = plt.cm.tab10.colors

    for i, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = box
        if class_colors and label in class_colors:
            color = class_colors[label]
        else:
            color = default_colors[hash(label) % len(default_colors)]

        ax.add_patch(patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor="none",
        ))
        text = f"{label} {scores[i]:.2f}" if scores is not None else label
        ax.text(
            x1, y1 - 5, text,
            fontsize=9, color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8),
        )

    ax.axis("off")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig
```

Notes:

- **Boxes are xyxy**; `patches.Rectangle` wants `(x, y), width, height`, hence the
  `x2 - x1` / `y2 - y1` conversion. Passing xywh boxes straight through is the most common bug here.
- **`facecolor="none"`** keeps the rectangle an outline; omitting it fills the box opaque.
- **Pass `class_colors`** for a stable project palette. The `hash(label)` fallback is not stable
  across Python processes (string hashing is salted), so do not rely on it for figures that
  must match between runs.
- **Label text sits above the box** (`y1 - 5`) with a rounded, semi-transparent background patch
  so it stays readable over any image content. Detections at the top edge will have their label
  clipped — clamp `y1 - 5` to a minimum of 0 if that matters.
- Use this instead of `cv2.rectangle` when the output is a notebook figure or a paper: text
  renders crisply and PDF/SVG output stays vector.
