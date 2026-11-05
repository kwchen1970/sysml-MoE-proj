"""Result figures in the layout of FlashMoE Figures 8-12.

The paper draws grouped bars, one panel per GPU count, with the series name,
shape, and an up/down arrow in the axes title. We only measured FlashMoE, so
each panel has that one hatched series. Baselines are not invented.
"""

import csv
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2] / "results" / "benchmark_confirmation"
OUT = ROOT

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "axes.linewidth": 0.8,
})

NOTE = (
    "FlashMoE only. Perlmutter A100, BF16 gated MLP, H=2048, k=2, capacity factor 1.0. "
    "Mean of 4 eager iterations after 1 warmup. "
    "1 GPU and 4 GPUs: A100-80GB, one node. 8 GPUs: A100-40GB across Slingshot-11. "
    "16 GPUs: mixed 40GB and 80GB. "
    "arXiv 2506.04667v3 is FP32 on H100s and includes baselines that were not run here."
)


def load(name):
    rows = []
    with (ROOT / name).open() as handle:
        for row in csv.DictReader(handle):
            for key in ("ngpu", "S", "H", "I", "E", "k"):
                row[key] = int(row[key])
            row["latency_ms"] = float(row["latency_ms"])
            rows.append(row)
    return rows


def mean_by(rows, keys):
    buckets = defaultdict(list)
    for row in rows:
        buckets[tuple(row[key] for key in keys)].append(row["latency_ms"])
    out = []
    for key, values in buckets.items():
        record = dict(zip(keys, key))
        record["latency_ms"] = sum(values) / len(values)
        out.append(record)
    return out


def lookup(rows, **want):
    hits = [row["latency_ms"] for row in rows if all(row[key] == value for key, value in want.items())]
    if len(hits) != 1:
        raise SystemExit(f"expected one row for {want}, found {len(hits)}")
    return hits[0]


def style(ax):
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", color="0.75", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.8)


def flash_bars(ax, labels, values, fmt="{:.1f}"):
    positions = list(range(len(labels)))
    bars = ax.bar(
        positions,
        values,
        width=0.62,
        color="#4C78D8",
        edgecolor="#1B3A6B",
        hatch="///",
        linewidth=0.5,
        label="FlashMoE",
    )
    ax.set_xticks(positions, labels)
    top = max(values) if values else 1
    ax.set_ylim(0, top * 1.28)
    for bar, value in zip(bars, values):
        ax.annotate(
            fmt.format(value),
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=7,
            xytext=(0, 1),
            textcoords="offset points",
        )
    ax.legend(loc="upper left", frameon=True, edgecolor="0.4")
    style(ax)
    return bars


def caption(fig, text, width=118):
    fig.text(0.01, 0.012, textwrap.fill(text, width), fontsize=7.5, va="bottom")


def save(fig, path):
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(path)


GPU_STYLE = {
    1: ("1 A100", "#4C78D8", "///", "#1B3A6B"),
    4: ("4 A100s", "#F58518", "\\\\\\", "#8C4A08"),
    8: ("8 A100s", "#54A24B", "...", "#1E5C28"),
    16: ("16 A100s", "#B279A2", "xx", "#5C3D56"),
}


def grouped(ax, labels, by_gpu, fmt="{:.1f}"):
    """One axes. by_gpu maps an ngpu count to its bar heights, aligned with labels."""
    ngpus = [ngpu for ngpu in (1, 4, 8) if ngpu in by_gpu]
    width = 0.8 / len(ngpus)
    centers = list(range(len(labels)))
    top = 1.0
    for index, ngpu in enumerate(ngpus):
        name, color, hatch, edge = GPU_STYLE[ngpu]
        offset = (index - (len(ngpus) - 1) / 2) * width
        positions = [center + offset for center in centers]
        values = by_gpu[ngpu]
        bars = ax.bar(
            positions,
            values,
            width=width * 0.92,
            color=color,
            edgecolor=edge,
            hatch=hatch,
            linewidth=0.4,
            label=name,
        )
        top = max(top, max(values))
        for bar, value in zip(bars, values):
            ax.annotate(
                fmt.format(value),
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=6.5,
                xytext=(0, 1),
                textcoords="offset points",
            )
    ax.set_xticks(centers, labels)
    ax.set_ylim(0, top * 1.22)
    ax.legend(loc="upper left", frameon=True, edgecolor="0.4")
    style(ax)


def count_chart(title, ylabel, xlabel, ngpus, values, path, caption_text, fmt="{:.1f}", ylim=None, legend_loc="upper left"):
    """One bar per GPU count, colored the same way as the overlaid sweeps."""
    fig, ax = plt.subplots(figsize=(5.15, 4.7))
    positions = list(range(len(ngpus)))
    top = 1.0
    for position, ngpu, value in zip(positions, ngpus, values):
        name, color, hatch, edge = GPU_STYLE[ngpu]
        bar = ax.bar(
            position,
            value,
            width=0.62,
            color=color,
            edgecolor=edge,
            hatch=hatch,
            linewidth=0.5,
            label=name,
        )
        top = max(top, value)
        ax.annotate(
            fmt.format(value),
            (bar[0].get_x() + bar[0].get_width() / 2, value),
            ha="center",
            va="bottom",
            fontsize=8,
            xytext=(0, 2),
            textcoords="offset points",
        )
    ax.set_xticks(positions, [str(ngpu) for ngpu in ngpus])
    ax.set_ylim(0, ylim if ylim is not None else top * 1.22)
    ax.legend(loc=legend_loc, frameon=True, edgecolor="0.4")
    style(ax)
    ax.set_title(title, fontsize=9, pad=6)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    fig.tight_layout(rect=(0, 0.22, 1, 1))
    caption(fig, caption_text, width=72)
    save(fig, path)


def one_chart(title, ylabel, xlabel, labels, by_gpu, path, caption_text, fmt="{:.1f}"):
    fig, ax = plt.subplots(figsize=(5.15, 4.7))
    grouped(ax, labels, by_gpu, fmt)
    ax.set_title(title, fontsize=9, pad=6)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    fig.tight_layout(rect=(0, 0.22, 1, 1))
    caption(fig, caption_text, width=72)
    save(fig, path)


def main():
    tokens = mean_by(load("bench_tokens.csv"), ["ngpu", "S", "H", "I", "E", "k"])
    experts = mean_by(load("bench_experts.csv"), ["ngpu", "S", "H", "I", "E", "k"])
    for row in tokens:
        if row["S"] == 4096 and row["I"] == 2048 and row["E"] == 32:
            experts.append(row)
    ffn = mean_by(load("bench_ffn.csv"), ["ngpu", "S", "H", "I", "E", "k"])
    for row in tokens:
        if row["ngpu"] in (1, 4) and row["S"] == 4096 and row["E"] == 32 and row["I"] == 2048:
            ffn.append(row)
    scale16 = lookup(mean_by(load("bench_scale.csv"), ["ngpu", "S", "H", "I", "E", "k"]), ngpu=16, S=4096, H=2048, I=2048, E=32, k=2)
    util = load("bench_util.csv")[0]

    token_labels = ["1K", "4K", "8K", "16K"]
    token_counts = [1024, 4096, 8192, 16384]

    def token_values(ngpu):
        return [lookup(tokens, ngpu=ngpu, S=s, H=2048, I=2048, E=32, k=2) for s in token_counts]

    one_chart(
        "Forward Latency | E = 32 | k = 2 | ↓ is better",
        "Runtime (ms)",
        "Number of Tokens",
        token_labels,
        {ngpu: token_values(ngpu) for ngpu in (1, 4, 8)},
        OUT / "latency_vs_tokens.png",
        "Forward latency as the number of tokens per GPU increases. " + NOTE,
    )

    expert_labels = ["8", "16", "32", "128"]
    expert_counts = [8, 16, 32, 128]

    def expert_values(ngpu):
        return [lookup(experts, ngpu=ngpu, S=4096, H=2048, I=2048, E=e, k=2) for e in expert_counts]

    one_chart(
        "Forward Latency | S = 4K | k = 2 | ↓ is better",
        "Runtime (ms)",
        "Number of Experts",
        expert_labels,
        {ngpu: expert_values(ngpu) for ngpu in (1, 4, 8)},
        OUT / "latency_vs_experts.png",
        "Forward latency as the number of experts increases. S=4096, not the paper's 16K. " + NOTE,
    )

    ffn_labels = ["1K", "2K", "4K", "8K"]
    ffn_dims = [1024, 2048, 4096, 8192]
    one_chart(
        "Forward Latency | S = 4K | E = 32 | k = 2 | ↓ is better",
        "Runtime (ms)",
        "FFN Intermediate Size",
        ffn_labels,
        {ngpu: [lookup(ffn, ngpu=ngpu, S=4096, H=2048, I=i, E=32, k=2) for i in ffn_dims] for ngpu in (1, 4)},
        OUT / "latency_vs_ffn.png",
        "Forward latency as the expert intermediate size increases. 8 GPUs were not swept on this axis. " + NOTE,
    )

    anchor = {ngpu: lookup(tokens, ngpu=ngpu, S=4096, H=2048, I=2048, E=32, k=2) for ngpu in (1, 4, 8)}
    anchor[16] = scale16
    latency = [anchor[n] for n in (1, 4, 8, 16)]
    throughput = [n * 4096 / (anchor[n] / 1e3) / 1e6 for n in (1, 4, 8, 16)]
    efficiency = [100.0 * anchor[1] / anchor[n] for n in (1, 4, 8, 16)]

    gpu_counts = [1, 4, 8, 16]
    count_chart(
        "Forward Latency | S = 4K | E = 32 | k = 2 | ↓ is better",
        "Runtime (ms)",
        "# of A100s",
        gpu_counts,
        latency,
        OUT / "latency_vs_gpus.png",
        "Forward latency as the number of GPUs increases, at fixed tokens per GPU. " + NOTE,
    )
    count_chart(
        "Overlap Efficiency | S = 4K | E = 32 | k = 2 | ↑ is better",
        "T(1) / T(N) (%)",
        "# of A100s",
        gpu_counts,
        efficiency,
        OUT / "weak_scaling_efficiency.png",
        "Weak scaling at fixed tokens per GPU. Efficiency is T(1)/T(N). The paper uses T(2)/T(N). " + NOTE,
        fmt="{:.0f}",
        ylim=125,
        legend_loc="upper right",
    )
    count_chart(
        "Forward Pass | S = 4K | E = 32 | k = 2 | ↑ is better",
        "Throughput (MTokens/s)",
        "# of A100s",
        gpu_counts,
        throughput,
        OUT / "throughput_vs_gpus.png",
        "Throughput = (tokens per GPU × GPUs) / latency, the paper's Figure 10 definition. " + NOTE,
        fmt="{:.2f}",
    )

    fig, ax = plt.subplots(figsize=(3.6, 3.55))
    flash_bars(ax, ["FlashMoE"], [float(util["util_sm_mean"])], fmt="{:.1f}")
    ax.set_title("Forward Pass | S = 4K | E = 32 | k = 2 | 1 A100")
    ax.set_ylabel("SM util (%)")
    ax.set_ylim(0, 110)
    ax.set_xlabel("")
    fig.tight_layout(rect=(0, 0.28, 1, 1))
    caption(
        fig,
        f"nvidia-smi dmon at 1 Hz over 800 iterations (peak {float(util['util_sm_max']):.0f}%). "
        "The paper's Figure 9 is Nsight SM-active, averaged over 100 iterations.",
        width=48,
    )
    save(fig, OUT / "sm_utilization.png")


if __name__ == "__main__":
    main()
