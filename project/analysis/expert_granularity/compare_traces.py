"""Compare validated 1-/4-GPU traces without treating ranks as repetitions."""
import argparse
import csv
import statistics as stats
from collections import defaultdict
from pathlib import Path
from intervals import load
from plotting import plt, lineplot, save_figure

STATES = ("task", "wait", "other")
TASKS = {"0": "GEMM0", "1": "GEMM1", "2": "Combine"}

def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("events", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    groups = load(a.events)
    rank_states = defaultdict(lambda: defaultdict(int))
    rank_tasks = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for key, events in groups.items():
        if key[7] != "processor":
            continue
        for event in events:
            duration = event["end_ns"] - event["start_ns"]
            rank_states[key[:5]][event["state"]] += duration
            if event["state"] == "task":
                kind = event.get("task_type", "unknown")
                rank_tasks[key[:5]][kind][0] += 1
                rank_tasks[key[:5]][kind][1] += duration
    if not rank_states:
        p.error("No processor events")
    a.output.mkdir(parents=True, exist_ok=True)
    rank_rows = []
    inv_states = defaultdict(lambda: defaultdict(int))
    inv_tasks = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    present_ranks = defaultdict(set)
    rank_points = defaultdict(list)
    for key, states in sorted(rank_states.items()):
        run, invocation, ngpu, experts, rank = key
        total = sum(states.values())
        row = dict(run_id=run, invocation=invocation, ngpu=ngpu, E=experts, rank=rank)
        for state in STATES:
            row[state + "_pct"] = 100 * states[state] / total
            inv_states[key[:4]][state] += states[state]
        rank_rows.append(row)
        rank_points[ngpu, experts, rank].append(row["wait_pct"])
        present_ranks[key[:4]].add(rank)
        for kind, (count, duration) in rank_tasks[key].items():
            inv_tasks[key[:4]][kind][0] += count
            inv_tasks[key[:4]][kind][1] += duration
    for key, ranks in present_ranks.items():
        if ranks != set(range(key[2])):
            raise ValueError(f"Missing ranks in trace invocation {key}: {ranks}")
    write_csv(a.output / "per_rank_metrics.csv", rank_rows)
    fractions = defaultdict(lambda: defaultdict(list))
    task_points = defaultdict(list)
    count_points = defaultdict(list)
    task_rows = []
    for key, states in sorted(inv_states.items()):
        run, invocation, ngpu, experts = key
        total = sum(states.values())
        for state in STATES:
            fractions[ngpu, experts][state].append(100 * states[state] / total)
        for kind, (count, duration) in sorted(inv_tasks[key].items()):
            # Weight all tasks equally within invocation; invocations are repetitions.
            mean_us = duration / count / 1000
            task_points[ngpu, experts, kind].append(mean_us)
            count_points[ngpu, experts, kind].append(count / ngpu)
            task_rows.append(dict(run_id=run, invocation=invocation, ngpu=ngpu, E=experts,
                                  task_type=kind, total_count=count,
                                  count_per_gpu=count / ngpu, mean_us=mean_us))
    write_csv(a.output / "task_metrics.csv", task_rows)
    summaries = []
    configs = sorted(fractions, key=lambda k: (k[1], k[0]))
    fig, ax = plt.subplots(figsize=(10, 5.8), layout="constrained")
    bottom = [0.] * len(configs)
    colors = {"task": "#2563eb", "wait": "#f59e0b", "other": "#9ca3af"}
    labels = {"task": "Task activity", "wait": "Mailbox waiting", "other": "Other"}
    for state in STATES:
        values = [stats.mean(fractions[k][state]) for k in configs]
        ax.bar(range(len(configs)), values, bottom=bottom, color=colors[state],
               label=labels[state], width=.7)
        bottom = [x + y for x, y in zip(bottom, values)]
    for ngpu, experts in configs:
        row = dict(ngpu=ngpu, E=experts, invocations=len(fractions[ngpu, experts]["wait"]))
        for state in STATES:
            values = fractions[ngpu, experts][state]
            row[state + "_mean_pct"] = stats.mean(values)
            row[state + "_sd_pct"] = stats.stdev(values) if len(values) > 1 else 0
        summaries.append(row)
    write_csv(a.output / "block_time_breakdown.csv", summaries)
    ax.set_xticks(range(len(configs)), [f"E={e}\n{ng} GPU" for ng, e in configs])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Recorded processor block-time (%)")
    ax.set_title("Processor activity by expert count and GPU count")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.15), ncol=3, frameon=False)
    save_figure(fig, a.output / "block_time_breakdown.png")

    def point(experts, values):
        return (experts, stats.mean(values), stats.stdev(values) if len(values) > 1 else 0)
    for filename, data, title, ylabel in [
        ("task_duration_comparison", task_points, "Task duration by expert and GPU count", "Mean recorded task duration (µs)"),
        ("task_count_comparison", count_points, "Task count per GPU by expert count", "Scheduled tasks per GPU per captured launch"),
    ]:
        curves = defaultdict(list)
        for (ngpu, experts, kind), values in sorted(data.items()):
            curves[f"{TASKS.get(kind, kind)}, {ngpu} GPU"].append(point(experts, values))
        lineplot(a.output / (filename + ".png"), curves, title, ylabel)
    curves = defaultdict(list)
    for (ngpu, experts, rank), values in sorted(rank_points.items()):
        if ngpu > 1:
            curves[f"{ngpu} GPU / rank {rank}"].append(point(experts, values))
    if curves:
        lineplot(a.output / "per_rank_waiting.png", curves,
                 "Rank-local processor waiting (mean ± sample SD across invocations)",
                 "Recorded processor waiting (%)")
    print(f"Saved comparison figures and CSVs to {a.output}")
    print("Ranks are not independent repetitions. Task counts are normalized per GPU.")
    print("Waiting is not an attribution of scheduler overhead.")

if __name__ == "__main__":
    main()
