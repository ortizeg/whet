# Confusion Matrix Visualization

Rendering an annotated, optionally row-normalized confusion matrix for classification results.

## Implementation

```python
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    normalize: bool = True,
    save_path: str | Path | None = None,
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
    figsize: tuple[int, int] = (10, 8),
) -> plt.Figure:
    """Plot an annotated confusion matrix, optionally row-normalized. Returns the figure."""
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)  # handle classes with zero samples

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted",
        ylabel="True",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(
                j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=10,
            )

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig
```

## Notes

- **Row normalization** (`cm.sum(axis=1, keepdims=True)`) turns counts into per-true-class
  recall, which is what you almost always want on an imbalanced dataset — raw counts make a
  dominant class visually swamp everything else.
- **`np.nan_to_num`** handles classes with zero samples in `y_true`, which would otherwise
  produce a row of NaNs and a blank stripe in the image.
- **Adaptive text color.** The `thresh = cm.max() / 2.0` comparison switches annotation text
  between white and black so labels stay readable on both ends of the colormap.
- **Rotate x tick labels** (`rotation=45, ha="right"`) as soon as class names are longer than a
  few characters, otherwise they overlap.
- **The format switches with normalization**: `".2f"` for fractions, `"d"` for integer counts.
- For many classes (>25), drop the per-cell text annotations entirely — they become unreadable.
  Keep the colormap and rely on the colorbar.
- Return the figure rather than calling `plt.show()`, so the caller can save, log to a tracker,
  or compose it into a larger report.
