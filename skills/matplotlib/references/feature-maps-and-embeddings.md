# Feature Maps and Embedding Plots

Visualizing what a network sees: CNN activation channels captured through forward hooks, and t-SNE/UMAP projections of learned embeddings.

## Contents

- [Feature Map Visualization](#feature-map-visualization)
- [Capturing Activations with Forward Hooks](#capturing-activations-with-forward-hooks)
- [t-SNE / UMAP Embedding Plots](#t-sne--umap-embedding-plots)

## Feature Map Visualization

```python
import torch


def plot_feature_maps(
    feature_map: torch.Tensor,     # (C, H, W) or (1, C, H, W)
    num_channels: int = 16,
    ncols: int = 8,
    save_path: str | Path | None = None,
    title: str = "Feature Maps",
) -> plt.Figure:
    """Visualize the first N channels of a CNN feature map. Returns the figure."""
    if feature_map.dim() == 4:
        feature_map = feature_map[0]

    fm = feature_map.detach().cpu().numpy()
    num_channels = min(num_channels, fm.shape[0])
    nrows = (num_channels + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    axes = np.atleast_2d(axes)
    for idx in range(nrows * ncols):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        if idx < num_channels:
            ax.imshow(fm[idx], cmap="viridis")
            ax.set_title(f"Ch {idx}", fontsize=8)
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig
```

Notes:

- **`.detach().cpu().numpy()`** is mandatory — plotting a tensor that still requires grad or
  lives on the GPU raises. Detaching also prevents the figure from pinning the autograd graph.
- **The `dim() == 4` check** takes the first item of a batch, so the same function works with
  raw hook output `(N, C, H, W)` and with a single sample `(C, H, W)`.
- **`num_channels` is clamped** to the real channel count, so a 32-channel layer does not index
  out of bounds when the default asks for more.
- Deep layers have hundreds of channels; the first 16 are an arbitrary sample. To find
  *interesting* channels, sort by activation variance or mean before slicing.
- `viridis` is perceptually uniform — prefer it over `jet` for activation magnitude.

## Capturing Activations with Forward Hooks

```python
# Capture activations via a forward hook, then plot:
activations: dict[str, torch.Tensor] = {}

def hook_fn(name):
    def hook(module, input, output):
        activations[name] = output
    return hook

model.layer3.register_forward_hook(hook_fn("layer3"))
model(input_batch)
plot_feature_maps(activations["layer3"], save_path="feature_maps.png")
```

- The closure over `name` is what lets one factory register hooks on many layers into one dict.
- `register_forward_hook` returns a handle — call `handle.remove()` when done, otherwise the
  hook keeps firing (and keeps a reference to the output tensor) for the rest of the process.
- Run under `torch.no_grad()` and in `model.eval()` mode for visualization.

## t-SNE / UMAP Embedding Plots

```python
def plot_embeddings(
    embeddings: np.ndarray,        # (N, D)
    labels: np.ndarray,            # (N,)
    class_names: list[str] | None = None,
    method: str = "tsne",          # "tsne" or "umap"
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (10, 8),
    title: str | None = None,
    perplexity: int = 30,
) -> plt.Figure:
    """Reduce embeddings to 2D and scatter by class. Returns the figure."""
    if method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    elif method == "umap":
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")

    coords = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=figsize)
    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    for i, label in enumerate(unique_labels):
        mask = labels == label
        name = class_names[label] if class_names else str(label)
        ax.scatter(coords[mask, 0], coords[mask, 1], c=[colors[i]], label=name, s=10, alpha=0.7)

    ax.legend(markerscale=3, fontsize=8, loc="best")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title(title or f"{method.upper()} Embedding Visualization")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig
```

Notes:

- **`random_state=42`** on both reducers — t-SNE and UMAP are stochastic, and without a fixed
  seed the same embeddings produce a visually different layout every run.
- **`perplexity` must be less than the number of samples** (sklearn enforces this) and controls
  the local/global balance; 5–50 is the useful range.
- **Distances between clusters in t-SNE are not meaningful** — only local neighborhood structure
  is. Do not read "these two classes are far apart" as a claim about the embedding space. UMAP
  preserves more global structure but the same caution applies.
- **`markerscale=3`** in the legend compensates for the tiny `s=10` scatter points, which would
  otherwise be invisible in the legend.
- `tab20` supports up to 20 distinct classes. Beyond that, colors repeat — switch to a
  continuous colormap or plot a subset of classes.
- t-SNE is O(N²)-ish; subsample to a few thousand points before plotting a large validation set.
