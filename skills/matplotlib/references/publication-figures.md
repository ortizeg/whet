# Publication-Quality Figures

Matplotlib settings and chart patterns for figures destined for a paper: small serif fonts, thin lines, single-column width, vector PDF output.

## Publication Style

Small serif fonts, thin lines, single-column width, PDF output.

```python
def setup_publication_style() -> None:
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        "figure.figsize": (3.5, 2.5),  # single-column width
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.format": "pdf",
        "savefig.bbox_inches": "tight",
        "savefig.pad_inches": 0.05,
        "font.size": 8,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "axes.linewidth": 0.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "lines.linewidth": 1.0,
        "lines.markersize": 3,
        "legend.fontsize": 7,
        "legend.framealpha": 0.8,
        "grid.linewidth": 0.3,
        "grid.alpha": 0.3,
        "text.usetex": False,
        "mathtext.fontset": "dejavuserif",
    })
```

Why these values:

- **`figure.figsize = (3.5, 2.5)`** is single-column width for a two-column paper. Sizing the
  figure correctly at creation time is what keeps font sizes honest — never scale a figure in
  LaTeX, that shrinks the text along with it.
- **8 pt serif body text** matches typical caption size, so labels read at the same scale as
  surrounding text.
- **Thin lines** (`axes.linewidth 0.5`, `lines.linewidth 1.0`, `grid.linewidth 0.3`) — default
  matplotlib weights look crude at print size.
- **PDF by default** keeps output vector, so it scales without pixelation.
- **`text.usetex = False`** with `mathtext.fontset = "dejavuserif"` gives LaTeX-like math
  rendering without requiring a TeX installation on the machine building figures. Flip
  `usetex` to `True` only if your build environment reliably has TeX.

## Grouped Comparison Bar Chart

```python
def plot_comparison_bar_chart(
    methods: list[str],
    metrics: dict[str, list[float]],
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Grouped bar chart comparing methods across metrics, with value labels."""
    setup_publication_style()
    n_metrics = len(metrics)
    x = np.arange(len(methods))
    width = 0.8 / n_metrics

    fig, ax = plt.subplots()
    colors = plt.cm.Set2(np.linspace(0, 0.8, n_metrics))
    for i, (metric_name, values) in enumerate(metrics.items()):
        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=metric_name, color=colors[i])
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=6,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.legend()
    ax.set_ylim(0, 100)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig
```

Notes:

- **`width = 0.8 / n_metrics`** leaves 20% of each x slot as a gap between method groups, and
  the `offset` expression centers the group on its tick regardless of how many metrics there are.
- **Value labels above bars** (`fontsize=6`) let a reader cite exact numbers without a
  companion table — worth the clutter in a results figure.
- **`set_ylim(0, 100)`** anchors the axis at zero. Truncated bar-chart axes exaggerate
  differences and are a reviewer red flag.
- `Set2` is a muted qualitative palette that survives grayscale printing better than `tab10`.
- Call `setup_publication_style()` before creating the figure — rcParams are read at artist
  creation time, so changing them afterwards has no effect on an existing figure.
