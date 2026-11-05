import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

source = Path(
    "project/results/expert_granularity/trace_1gpu/events.csv"
)
output = Path(
    "project/results/expert_granularity/trace_1gpu"
)
output.mkdir(parents=True, exist_ok=True)

# Use raw event intervals, not the already-aggregated task CSV.
# Exclude OS lanes because their "other" state is OS lifetime.
durations = defaultdict(lambda: defaultdict(int))
dropped = set()

with source.open() as f:
    for row in csv.DictReader(f):
        if row["role"] != "processor":
            continue

        key = (
            row["run_id"],
            int(row["invocation"]),
            int(row["ngpu"]),
            int(row["E"]),
            int(row["rank"]),
        )

        state = row["state"]
        if state not in ("task", "wait", "other"):
            continue

        duration = int(row["end_ns"]) - int(row["start_ns"])
        if duration < 0:
            raise ValueError("Negative event duration")

        durations[key][state] += duration

        if int(row["dropped_events"]) > 0:
            dropped.add(key)

if dropped:
    raise RuntimeError(
        f"{len(dropped)} traces have dropped events; "
        "do not plot incomplete traces."
    )

# Calculate percentages separately for each invocation.
samples = defaultdict(lambda: defaultdict(list))

for key, states in durations.items():
    _, invocation, ngpu, experts, rank = key
    total = sum(states.values())

    if total == 0:
        continue

    for state in ("task", "wait", "other"):
        samples[(ngpu, experts)][state].append(
            100 * states[state] / total
        )

experts = sorted({
    e for (g, e) in samples if g == 1
})

if not experts:
    raise RuntimeError("No 1-GPU processor traces found")

states = ["task", "wait", "other"]
labels = ["Task activity", "Mailbox waiting", "Other"]
colors = ["#2563eb", "#f59e0b", "#9ca3af"]

fig, ax = plt.subplots(figsize=(8.5, 5.2))
x = np.arange(len(experts))
bottom = np.zeros(len(experts))

summary = []

for state, label, color in zip(states, labels, colors):
    values = [
        mean(samples[(1, e)][state]) for e in experts
    ]

    ax.bar(
        x, values, bottom=bottom,
        color=color, label=label, width=0.58
    )

    for i, value in enumerate(values):
        if value >= 4:
            ax.text(
                x[i], bottom[i] + value / 2,
                f"{value:.1f}%",
                ha="center", va="center",
                fontsize=10,
                color="white" if state == "task" else "black"
            )

    bottom += np.array(values)

for e in experts:
    row = {"experts": e, "gpu_count": 1}
    for state in states:
        values = samples[(1, e)][state]
        row[state + "_mean_pct"] = mean(values)
        row[state + "_sd_pct"] = (
            stdev(values) if len(values) > 1 else 0
        )
    summary.append(row)

ax.set_xticks(x, [str(e) for e in experts])
ax.set_xlabel("Number of Experts")
ax.set_ylabel("Fraction of Recorded Processor Block-Time (%)")
ax.set_title("FlashMoE GPU Block-Time Breakdown (1 GPU)")
ax.set_ylim(0, 100)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15),
          ncol=3, frameon=False)
ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig(output / "block_time_breakdown.png",
            dpi=250, bbox_inches="tight")
fig.savefig(output / "block_time_breakdown.pdf",
            bbox_inches="tight")

with (output / "block_time_breakdown.csv").open(
    "w", newline=""
) as f:
    writer = csv.DictWriter(f, fieldnames=summary[0].keys())
    writer.writeheader()
    writer.writerows(summary)

for row in summary:
    print(
        f"E={row['experts']}: "
        f"task={row['task_mean_pct']:.2f}%, "
        f"wait={row['wait_mean_pct']:.2f}%, "
        f"other={row['other_mean_pct']:.2f}%"
    )

print("Saved block_time_breakdown.png/.pdf/.csv")