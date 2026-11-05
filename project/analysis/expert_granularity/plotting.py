"""Matplotlib figures for headless jobs; export PNG, SVG and PDF together."""
from pathlib import Path
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ["#2563eb", "#dc2626", "#059669", "#9333ea"]
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

def save_figure(fig, path):
    """The requested filename supplies the stem; always write all formats."""
    stem = Path(path).with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    try:
        for suffix in (".png", ".svg", ".pdf"):
            fig.savefig(stem.with_suffix(suffix), dpi=300, bbox_inches="tight",
                        facecolor="white")
    finally:
        plt.close(fig)

def lineplot(path, groups, title, ylabel):
    # groups: label -> [(expert count, mean, sample standard deviation)]
    xs = sorted({x for points in groups.values() for x, y, std in points})
    if not xs:
        raise ValueError("Cannot plot empty measurements")
    position = {x: i for i, x in enumerate(xs)}
    fig, ax = plt.subplots(figsize=(10, 5.8), layout="constrained")
    for i, (label, points) in enumerate(groups.items()):
        points = sorted(points)
        ax.errorbar([position[x] for x, y, std in points],
                    [y for x, y, std in points],
                    yerr=[std for x, y, std in points],
                    marker="o", capsize=4, linewidth=1.8,
                    color=COLORS[i % len(COLORS)], label=label)
    ax.set_xticks(range(len(xs)), [str(x) for x in xs])
    ax.set_xlabel("Number of Experts")
    ax.set_ylabel(ylabel)
    ax.set_title(textwrap.fill(title, 85), pad=12)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.18),
              frameon=False, fontsize=9)
    save_figure(fig, path)
