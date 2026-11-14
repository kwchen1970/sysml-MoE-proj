"""Compare single-expert affinity (SCHED) vs Ampere gated bK=64/pS=1 (BK64).

Input: project/results/bk64_ab/compare.csv from interleaved A/B job 61293689.
"""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "results" / "bk64_ab"
OUT = ROOT

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

SERIES = {
    "SCHED": ("SCHED (bK=32, pS=2)", "#4C78D8", "///", "#1B3A6B"),
    "BK64": ("BK64 (bK=64, pS=1)", "#54A24B", "\\\\\\", "#1E5C28"),
}

NOTE = (
    "BF16 gated MLP on Perlmutter A100-SXM4-40GB. "
    "Interleaved A/B job 61293689: mean of 3 rounds, 2 warmup + 16 timed iters. "
    "SCHED is single-expert affinity on exp/scheduler-throughput; "
    "BK64 is the same scheduler with Ampere gated bK=64 / one K-stage. "
    "Error bars are min–max across rounds."
)


def load(path: Path):
    rows = []
    with path.open() as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "tag": row["tag"],
                    "round": int(row["round"]),
                    "S": int(row["S"]),
                    "H": int(row["H"]),
                    "I": int(row["I"]),
                    "E": int(row["E"]),
                    "k": int(row["k"]),
                    "latency_ms": float(row["latency_ms"]),
                }
            )
    return rows


def group(rows):
    buckets = defaultdict(list)
    for row in rows:
        key = (row["tag"], row["S"], row["H"], row["I"], row["E"], row["k"])
        buckets[key].append(row["latency_ms"])
    out = {}
    for key, values in buckets.items():
        out[key] = {
            "mean": statistics.mean(values),
            "lo": min(values),
            "hi": max(values),
            "n": len(values),
        }
    return out


def style(ax):
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", color="0.75", linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.8)


def paired_bars(ax, labels, sched, bk64, ylabel, title):
    x = np.arange(len(labels))
    width = 0.36
    for offset, tag, series in (
        (-width / 2, "SCHED", sched),
        (width / 2, "BK64", bk64),
    ):
        name, color, hatch, edge = SERIES[tag]
        means = [s["mean"] for s in series]
        yerr = np.array(
            [[s["mean"] - s["lo"] for s in series], [s["hi"] - s["mean"] for s in series]]
        )
        bars = ax.bar(
            x + offset,
            means,
            width=width,
            color=color,
            edgecolor=edge,
            hatch=hatch,
            linewidth=0.5,
            label=name,
            yerr=yerr,
            error_kw={"ecolor": "0.2", "capsize": 2, "linewidth": 0.7},
        )
        for bar, value in zip(bars, means):
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
    top = max(max(s["hi"] for s in sched), max(s["hi"] for s in bk64))
    ax.set_ylim(0, top * 1.28)


def delta_bars(ax, labels, sched, bk64, title):
    x = np.arange(len(labels))
    deltas = [100.0 * (b["mean"] - s["mean"]) / s["mean"] for s, b in zip(sched, bk64)]
    colors = ["#54A24B" if d < 0 else "#E45756" for d in deltas]
    bars = ax.bar(x, deltas, color=colors, edgecolor="0.2", linewidth=0.4)
    ax.axhline(0, color="0.2", linewidth=0.8)
    for bar, value in zip(bars, deltas):
        va = "top" if value < 0 else "bottom"
        offset = -3 if value < 0 else 3
        ax.annotate(
            f"{value:+.1f}%",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va=va,
            fontsize=7,
            xytext=(0, offset),
            textcoords="offset points",
        )
    ax.set_xticks(x, labels)
    ax.set_ylabel("Latency change vs SCHED (%)")
    ax.set_title(title)
    style(ax)
    lim = max(8.0, max(abs(d) for d in deltas) * 1.35)
    ax.set_ylim(-lim, lim)


def lookup(stats, tag, **want):
    for (t, S, H, I, E, k), value in stats.items():
        if t != tag:
            continue
        if all(want[key] == locals_map(S, H, I, E, k)[key] for key in want):
            return value
    raise KeyError(want)


def locals_map(S, H, I, E, k):
    return {"S": S, "H": H, "I": I, "E": E, "k": k}


def series_for(stats, tag, specs):
    out = []
    for spec in specs:
        out.append(lookup(stats, tag, **spec))
    return out


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(path)


def main():
    stats = group(load(ROOT / "compare.csv"))

    token_specs = [
        {"S": 1024, "H": 2048, "I": 2048, "E": 32, "k": 2},
        {"S": 4096, "H": 2048, "I": 2048, "E": 32, "k": 2},
        {"S": 8192, "H": 2048, "I": 2048, "E": 32, "k": 2},
        {"S": 16384, "H": 2048, "I": 2048, "E": 32, "k": 2},
    ]
    token_labels = ["1024", "4096", "8192", "16384"]
    ffn_specs = [
        {"S": 4096, "H": 2048, "I": 1024, "E": 32, "k": 2},
        {"S": 4096, "H": 2048, "I": 2048, "E": 32, "k": 2},
        {"S": 4096, "H": 2048, "I": 4096, "E": 32, "k": 2},
        {"S": 4096, "H": 2048, "I": 8192, "E": 32, "k": 2},
    ]
    ffn_labels = ["1024", "2048", "4096", "8192"]
    expert_specs = [
        {"S": 4096, "H": 2048, "I": 2048, "E": 8, "k": 2},
        {"S": 4096, "H": 2048, "I": 2048, "E": 16, "k": 2},
        {"S": 4096, "H": 2048, "I": 2048, "E": 32, "k": 2},
        {"S": 4096, "H": 2048, "I": 2048, "E": 128, "k": 2},
    ]
    expert_labels = ["8", "16", "32", "128"]
    anchor_specs = [{"S": 4096, "H": 2048, "I": 1408, "E": 16, "k": 6}]

    # Tokens
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    sched = series_for(stats, "SCHED", token_specs)
    bk64 = series_for(stats, "BK64", token_specs)
    paired_bars(
        axes[0],
        token_labels,
        sched,
        bk64,
        "Forward latency (ms)",
        "Latency vs tokens (H=2048, I=2048, E=32, k=2)",
    )
    axes[0].set_xlabel("Sequence length S")
    delta_bars(axes[1], token_labels, sched, bk64, "Speedup vs SCHED (negative is faster)")
    axes[1].set_xlabel("Sequence length S")
    fig.text(0.01, -0.02, NOTE, ha="left", va="top", fontsize=7, wrap=True)
    save(fig, "latency_vs_tokens.png")

    # FFN
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    sched = series_for(stats, "SCHED", ffn_specs)
    bk64 = series_for(stats, "BK64", ffn_specs)
    paired_bars(
        axes[0],
        ffn_labels,
        sched,
        bk64,
        "Forward latency (ms)",
        "Latency vs FFN width (S=4096, H=2048, E=32, k=2)",
    )
    axes[0].set_xlabel("Intermediate size I")
    delta_bars(axes[1], ffn_labels, sched, bk64, "Speedup vs SCHED (negative is faster)")
    axes[1].set_xlabel("Intermediate size I")
    fig.text(0.01, -0.02, NOTE, ha="left", va="top", fontsize=7, wrap=True)
    save(fig, "latency_vs_ffn.png")

    # Experts
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    sched = series_for(stats, "SCHED", expert_specs)
    bk64 = series_for(stats, "BK64", expert_specs)
    paired_bars(
        axes[0],
        expert_labels,
        sched,
        bk64,
        "Forward latency (ms)",
        "Latency vs experts (S=4096, H=2048, I=2048, k=2)",
    )
    axes[0].set_xlabel("Expert count E")
    delta_bars(axes[1], expert_labels, sched, bk64, "Speedup vs SCHED (negative is faster)")
    axes[1].set_xlabel("Expert count E")
    fig.text(0.01, -0.02, NOTE, ha="left", va="top", fontsize=7, wrap=True)
    save(fig, "latency_vs_experts.png")

    # Anchor highlight
    sched = series_for(stats, "SCHED", anchor_specs)[0]
    bk64 = series_for(stats, "BK64", anchor_specs)[0]
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    paired_bars(
        ax,
        ["anchor\nS=4096 I=1408 E=16 k=6"],
        [sched],
        [bk64],
        "Forward latency (ms)",
        f"Project anchor: {100*(bk64['mean']-sched['mean'])/sched['mean']:+.1f}%",
    )
    ax.set_xlabel("")
    fig.text(0.01, -0.02, NOTE, ha="left", va="top", fontsize=7, wrap=True)
    save(fig, "latency_anchor.png")

    # Summary delta strip across all comparable shapes used above
    all_specs = token_specs + [
        {"S": 4096, "H": 2048, "I": 1024, "E": 32, "k": 2},
        {"S": 4096, "H": 2048, "I": 4096, "E": 32, "k": 2},
        {"S": 4096, "H": 2048, "I": 8192, "E": 32, "k": 2},
        {"S": 4096, "H": 2048, "I": 2048, "E": 8, "k": 2},
        {"S": 4096, "H": 2048, "I": 2048, "E": 16, "k": 2},
        {"S": 4096, "H": 2048, "I": 2048, "E": 128, "k": 2},
        {"S": 4096, "H": 2048, "I": 1408, "E": 16, "k": 6},
    ]
    labels = [
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
    # keep label list aligned with all_specs without duplicating S4096 paper twice
    all_specs = [
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
    sched = series_for(stats, "SCHED", all_specs)
    bk64 = series_for(stats, "BK64", all_specs)
    fig, ax = plt.subplots(figsize=(10.5, 3.8))
    delta_bars(ax, labels, sched, bk64, "BK64 vs SCHED across A/B shapes")
    ax.set_xlabel("Shape")
    fig.text(0.01, -0.08, NOTE, ha="left", va="top", fontsize=7, wrap=True)
    save(fig, "delta_summary.png")


if __name__ == "__main__":
    main()
