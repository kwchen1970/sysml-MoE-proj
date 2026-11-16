#!/usr/bin/env python3
"""Plot interleaved A/B compare.csv (from parse_keep_gate) into PNG/PDF.

Writes latency + delta panels and a summary strip. Designed for Project media/:

  python plot_ab.py path/to/compare.csv --cand HYB17 --base HYB2 \\
      --out-dir /path/to/media/hyb17_vs_hyb2 --title 'HYB17 vs HYB2'
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "figure.dpi": 140,
    }
)

PALETTE = [
    ("#4C78D8", "#1B3A6B", "///"),
    ("#54A24B", "#1E5C28", "\\\\\\"),
    ("#E45756", "#7A1F1F", "xxx"),
    ("#F58518", "#7A3F00", "..."),
    ("#B279A2", "#4A2A55", "+++"),
]


def load_compare(path: Path, tags: list[str]) -> list[dict]:
    rows = []
    with path.open() as handle:
        for row in csv.DictReader(handle):
            item = {
                "S": int(row["S"]),
                "H": int(row["H"]),
                "I": int(row["I"]),
                "E": int(row["E"]),
                "k": int(row["k"]),
            }
            for tag in tags:
                raw = row.get(tag, "")
                item[tag] = float(raw) if raw not in ("", None) else float("nan")
            rows.append(item)
    return rows


def style(ax):
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", color="0.75", linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.8)


def pick(rows, **want):
    for row in rows:
        if all(row[k] == v for k, v in want.items()):
            return row
    raise KeyError(want)


def paired_bars(ax, labels, series, ylabel, title):
    x = np.arange(len(labels))
    n = len(series)
    width = min(0.8 / n, 0.36)
    offsets = (np.arange(n) - (n - 1) / 2.0) * width
    for offset, (name, values, color, edge, hatch) in zip(offsets, series):
        bars = ax.bar(
            x + offset,
            values,
            width=width * 0.95,
            color=color,
            edgecolor=edge,
            hatch=hatch,
            linewidth=0.5,
            label=name,
        )
        for bar, value in zip(bars, values):
            if value == value:
                ax.annotate(
                    f"{value:.2f}",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    xytext=(0, 2),
                    textcoords="offset points",
                )
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper left", frameon=True, edgecolor="0.4")
    style(ax)
    finite = [v for _, vals, *_ in series for v in vals if v == v]
    ax.set_ylim(0, (max(finite) if finite else 1.0) * 1.28)


def delta_bars(ax, labels, deltas, title, ylabel):
    x = np.arange(len(labels))
    colors = ["#54A24B" if d < 0 else "#E45756" for d in deltas]
    bars = ax.bar(x, deltas, color=colors, edgecolor="0.2", linewidth=0.4)
    ax.axhline(0, color="0.2", linewidth=0.8)
    for bar, value in zip(bars, deltas):
        va = "top" if value < 0 else "bottom"
        off = -3 if value < 0 else 3
        ax.annotate(
            f"{value:+.1f}%",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va=va,
            fontsize=7,
            xytext=(0, off),
            textcoords="offset points",
        )
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    style(ax)
    lim = max(8.0, max((abs(d) for d in deltas), default=1.0) * 1.35)
    ax.set_ylim(-lim, lim)


def save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(path)


def series_pack(rows, tags, specs):
    packed = []
    for i, tag in enumerate(tags):
        color, edge, hatch = PALETTE[i % len(PALETTE)]
        vals = []
        for spec in specs:
            row = pick(rows, **spec)
            vals.append(row[tag])
        packed.append((tag, vals, color, edge, hatch))
    return packed


def deltas(rows, cand, base, specs):
    out = []
    for spec in specs:
        row = pick(rows, **spec)
        b = row[base]
        c = row[cand]
        out.append(100.0 * (c / b - 1.0) if b else float("nan"))
    return out


def geo_delta(rows, cand, base) -> float:
    ratios = []
    for row in rows:
        b = row[base]
        c = row[cand]
        if b == b and c == c and b > 0:
            ratios.append(c / b)
    if not ratios:
        return float("nan")
    return 100.0 * (math.exp(sum(math.log(r) for r in ratios) / len(ratios)) - 1.0)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("compare_csv", type=Path)
    p.add_argument("--cand", required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--tags", nargs="*", default=None, help="Tags to draw (default: base cand)")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--title", default="")
    p.add_argument("--note", default="")
    args = p.parse_args()

    tags = args.tags or [args.base, args.cand]
    # de-dupe preserve order
    seen = set()
    tags = [t for t in tags if not (t in seen or seen.add(t))]
    rows = load_compare(args.compare_csv, tags)
    note = args.note or (
        f"{args.title or (args.cand + ' vs ' + args.base)}. "
        f"Geo Δ {geo_delta(rows, args.cand, args.base):+.2f}% (negative = faster cand)."
    )

    token_specs = [
        {"S": 1024, "H": 2048, "I": 2048, "E": 32, "k": 2},
        {"S": 4096, "H": 2048, "I": 2048, "E": 32, "k": 2},
        {"S": 8192, "H": 2048, "I": 2048, "E": 32, "k": 2},
        {"S": 16384, "H": 2048, "I": 2048, "E": 32, "k": 2},
    ]
    ffn_specs = [
        {"S": 4096, "H": 2048, "I": 1024, "E": 32, "k": 2},
        {"S": 4096, "H": 2048, "I": 2048, "E": 32, "k": 2},
        {"S": 4096, "H": 2048, "I": 4096, "E": 32, "k": 2},
        {"S": 4096, "H": 2048, "I": 8192, "E": 32, "k": 2},
    ]
    expert_specs = [
        {"S": 4096, "H": 2048, "I": 2048, "E": 8, "k": 2},
        {"S": 4096, "H": 2048, "I": 2048, "E": 16, "k": 2},
        {"S": 4096, "H": 2048, "I": 2048, "E": 32, "k": 2},
        {"S": 4096, "H": 2048, "I": 2048, "E": 128, "k": 2},
    ]
    summary_specs = [
        {"S": 1024, "H": 2048, "I": 2048, "E": 32, "k": 2},
        {"S": 4096, "H": 2048, "I": 2048, "E": 32, "k": 2},
        {"S": 8192, "H": 2048, "I": 2048, "E": 32, "k": 2},
        {"S": 16384, "H": 2048, "I": 2048, "E": 32, "k": 2},
        {"S": 4096, "H": 2048, "I": 1024, "E": 32, "k": 2},
        {"S": 4096, "H": 2048, "I": 4096, "E": 32, "k": 2},
        {"S": 4096, "H": 2048, "I": 8192, "E": 32, "k": 2},
        {"S": 4096, "H": 2048, "I": 2048, "E": 8, "k": 2},
        {"S": 4096, "H": 2048, "I": 2048, "E": 16, "k": 2},
        {"S": 4096, "H": 2048, "I": 2048, "E": 128, "k": 2},
        {"S": 4096, "H": 2048, "I": 1408, "E": 16, "k": 6},
    ]
    summary_labels = [
        "S1024",
        "S4096",
        "S8192",
        "S16384",
        "I1024",
        "I4096",
        "I8192",
        "E8",
        "E16",
        "E128",
        "anchor",
    ]

    # Tokens
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    paired_bars(
        axes[0],
        ["1024", "4096", "8192", "16384"],
        series_pack(rows, tags, token_specs),
        "Forward latency (ms)",
        "Latency vs tokens",
    )
    axes[0].set_xlabel("Sequence length S")
    delta_bars(
        axes[1],
        ["1024", "4096", "8192", "16384"],
        deltas(rows, args.cand, args.base, token_specs),
        f"{args.cand} vs {args.base} (neg = faster)",
        f"Latency change vs {args.base} (%)",
    )
    axes[1].set_xlabel("Sequence length S")
    fig.text(0.01, -0.02, note, ha="left", va="top", fontsize=7, wrap=True)
    save(fig, args.out_dir, "latency_vs_tokens.png")

    # FFN
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    paired_bars(
        axes[0],
        ["1024", "2048", "4096", "8192"],
        series_pack(rows, tags, ffn_specs),
        "Forward latency (ms)",
        "Latency vs FFN width",
    )
    axes[0].set_xlabel("Intermediate size I")
    delta_bars(
        axes[1],
        ["1024", "2048", "4096", "8192"],
        deltas(rows, args.cand, args.base, ffn_specs),
        f"{args.cand} vs {args.base} (neg = faster)",
        f"Latency change vs {args.base} (%)",
    )
    axes[1].set_xlabel("Intermediate size I")
    fig.text(0.01, -0.02, note, ha="left", va="top", fontsize=7, wrap=True)
    save(fig, args.out_dir, "latency_vs_ffn.png")

    # Experts
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    paired_bars(
        axes[0],
        ["8", "16", "32", "128"],
        series_pack(rows, tags, expert_specs),
        "Forward latency (ms)",
        "Latency vs experts",
    )
    axes[0].set_xlabel("Expert count E")
    delta_bars(
        axes[1],
        ["8", "16", "32", "128"],
        deltas(rows, args.cand, args.base, expert_specs),
        f"{args.cand} vs {args.base} (neg = faster)",
        f"Latency change vs {args.base} (%)",
    )
    axes[1].set_xlabel("Expert count E")
    fig.text(0.01, -0.02, note, ha="left", va="top", fontsize=7, wrap=True)
    save(fig, args.out_dir, "latency_vs_experts.png")

    # Anchor
    anchor_spec = [{"S": 4096, "H": 2048, "I": 1408, "E": 16, "k": 6}]
    d_anchor = deltas(rows, args.cand, args.base, anchor_spec)[0]
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    paired_bars(
        ax,
        ["anchor\nS=4096 I=1408 E=16 k=6"],
        series_pack(rows, tags, anchor_spec),
        "Forward latency (ms)",
        f"Project anchor: {d_anchor:+.1f}% vs {args.base}",
    )
    fig.text(0.01, -0.02, note, ha="left", va="top", fontsize=7, wrap=True)
    save(fig, args.out_dir, "latency_anchor.png")

    # Summary strip
    fig, ax = plt.subplots(figsize=(10.5, 3.8))
    delta_bars(
        ax,
        summary_labels,
        deltas(rows, args.cand, args.base, summary_specs),
        f"{args.cand} vs {args.base} across A/B shapes (geo {geo_delta(rows, args.cand, args.base):+.2f}%)",
        f"Latency change vs {args.base} (%)",
    )
    ax.set_xlabel("Shape")
    fig.text(0.01, -0.08, note, ha="left", va="top", fontsize=7, wrap=True)
    save(fig, args.out_dir, "delta_summary.png")

    # Convenience copies with legacy hyb-style names
    for src, dst in (
        ("latency_vs_tokens.png", "latency.png"),
        ("delta_summary.png", "delta.png"),
        ("latency_anchor.png", "summary.png"),
    ):
        s = args.out_dir / src
        d = args.out_dir / dst
        if s.exists():
            d.write_bytes(s.read_bytes())
            pdf = s.with_suffix(".pdf")
            if pdf.exists():
                d.with_suffix(".pdf").write_bytes(pdf.read_bytes())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
