import csv
from collections import defaultdict
from statistics import mean, stdev
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

source = Path("project/results/expert_granularity/trace_1gpu/task_durations.csv")
output = Path("project/results/expert_granularity/trace_1gpu/task_duration_comparison.png")

data = defaultdict(list)
with source.open() as f:
    for row in csv.DictReader(f):
        key = (int(row["E"]), int(row["task_type"]))
        data[key].append(float(row["mean_us"]))

experts = [16, 64, 128]
operations = ["GEMM0", "GEMM1", "Combine"]
x = np.arange(len(experts))
width = 0.25

fig, ax = plt.subplots(figsize=(9, 5.5))

for task_id, name in enumerate(operations):
    values = [mean(data[(e, task_id)]) for e in experts]
    errors = [stdev(data[(e, task_id)]) for e in experts]
    ax.bar(x + (task_id - 1) * width, values, width,
           yerr=errors, capsize=4, label=name)

ax.set_xticks(x)
ax.set_xticklabels(experts)
ax.set_xlabel("Number of experts")
ax.set_ylabel("Mean task duration (microseconds)")
ax.set_title("FlashMoE Task Duration by Expert Count (1 GPU)")
ax.legend()
ax.grid(axis="y", alpha=0.2)
fig.tight_layout()
fig.savefig(output, dpi=200)
fig.savefig(output.with_suffix(".pdf"))
print(f"Saved {output}")
