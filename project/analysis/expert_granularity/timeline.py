"""Plot real device intervals with Matplotlib; not kernel-level Nsight data."""
import argparse
import math
from pathlib import Path
from intervals import load
from plotting import plt, save_figure
from matplotlib.patches import Patch

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("events", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--invocation", required=True)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--max-processors", type=int, default=12)
    p.add_argument("--duration-us", type=float,
                   help="Shared x-axis span for comparable panels; must cover this trace")
    a = p.parse_args()
    if not 1 <= a.max_processors <= 16:
        p.error("max-processors must be 1..16 for a readable figure")
    groups = load(a.events)
    selected = [(key, rows) for key, rows in groups.items()
                if key[0] == a.run_id and key[1] == a.invocation and key[4] == a.rank]
    selected.sort(key=lambda pair: (pair[0][7] == "processor", pair[0][5], pair[0][7]))
    lanes = []
    processors = 0
    for key, rows in selected:
        if key[7] == "processor":
            if processors >= a.max_processors:
                continue
            processors += 1
        lanes.append((key, rows))
    if not lanes:
        p.error("No lanes match the requested run/invocation/rank")
    start = min(r["window_start_ns"] for key, rows in lanes for r in rows)
    stop = max(r["window_end_ns"] for key, rows in lanes for r in rows)
    span_us = (stop - start) / 1000
    if a.duration_us is not None:
        if not math.isfinite(a.duration_us) or a.duration_us <= 0 or a.duration_us < span_us:
            p.error("duration-us must be finite, positive and cover the full selected window")
        span_us = a.duration_us
    colors = {"wait": "#f59e0b", "task": "#2563eb", "other": "#9ca3af"}
    fig, ax = plt.subplots(figsize=(11, max(4, .36 * len(lanes) + 2)),
                           layout="constrained")
    labels = []
    for i, (key, rows) in enumerate(lanes):
        labels.append(f"{key[7]} B{key[5]} SM{key[6]}")
        # Subtract integer timestamps BEFORE converting to plotting units.
        for state in sorted({r["state"] for r in rows}):
            intervals = [((r["start_ns"] - start) / 1000,
                          (r["end_ns"] - r["start_ns"]) / 1000)
                         for r in rows if r["state"] == state]
            ax.broken_barh(intervals, (i - .35, .7),
                           facecolors=colors.get(state, "#059669"), linewidth=0)
    ax.set_yticks(range(len(lanes)), labels)
    ax.set_ylim(len(lanes) - .5, -.5)
    ax.set_xlim(0, span_us)
    ax.set_xlabel("Time since local window start (µs)")
    ax.set_ylabel("Block / recorded SM")
    ax.set_title(f"E={lanes[0][0][3]}, rank={a.rank}; selected block lanes", pad=12)
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[
        Patch(facecolor=colors["task"], label="Task (includes transfers/signaling)"),
        Patch(facecolor=colors["wait"], label="Mailbox waiting"),
        Patch(facecolor=colors["other"], label="Other / OS lifetime"),
    ], loc="upper center", bbox_to_anchor=(.5, -.17), frameon=False, fontsize=9)
    save_figure(fig, a.output)

if __name__ == "__main__":
    main()
