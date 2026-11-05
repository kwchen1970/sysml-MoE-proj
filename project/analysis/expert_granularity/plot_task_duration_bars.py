"""Grouped task-duration bars from invocation-level v2 trace summaries."""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from plotting import plt, save_figure

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,default=Path("project/results/expert_granularity/study_1vs4_v2/trace/task_metrics.csv"))
    p.add_argument("--gpus",type=int,choices=[1,4],default=4)
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    samples=defaultdict(list)
    seen=set()
    with a.input.open() as f:
        for r in csv.DictReader(f):
            if int(r["ngpu"])!=a.gpus:continue
            key=(int(r["E"]),int(r["task_type"]))
            identity=(r["run_id"],r["invocation"],key)
            if identity in seen:raise ValueError("Duplicate invocation/task summary")
            seen.add(identity)
            value=float(r["mean_us"])
            if not math.isfinite(value) or value<=0:raise ValueError("Invalid task duration")
            samples[key].append(value)
    experts=sorted({key[0] for key in samples})
    if not experts: p.error("No matching GPU configuration")
    if any(len(samples[e,kind])<2 for e in experts for kind in range(3)):
        p.error("Need at least two invocation means for every expert count and task type")
    fig,ax=plt.subplots(figsize=(8,4.8))
    width=.25
    rows=[]
    for kind,label,color in [(0,"GEMM0","#1f77b4"),(1,"GEMM1","#ff7f0e"),(2,"Combine","#2ca02c")]:
        values=[mean(samples[e,kind]) for e in experts]
        errors=[stdev(samples[e,kind]) for e in experts]
        ax.bar([i+(kind-1)*width for i in range(len(experts))],values,width,
               yerr=errors,capsize=4,label=label,color=color)
        for e,value,error in zip(experts,values,errors):
            rows.append(dict(ngpu=a.gpus,E=e,task_type=kind,task_name=label,
                             invocations=len(samples[e,kind]),mean_us=value,sd_us=error))
    ax.set_xticks(range(len(experts)),[str(e) for e in experts])
    ax.set_xlabel("Number of experts")
    ax.set_ylabel("Mean recorded task duration (µs)")
    ax.set_title(f"FlashMoE Task Duration by Expert Count ({a.gpus} GPU"+("s)" if a.gpus!=1 else ")"))
    ax.grid(axis="y",alpha=.2)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right")
    ax.set_ylim(bottom=0)
    ax.text(.5,-.20,"Post-warmup traces. Bars: mean across runs; error bars: sample SD.\n"
            "Each run pools tasks across participating GPUs. Task time includes transfers/signaling.",
            transform=ax.transAxes,ha="center",va="top",fontsize=8)
    fig.tight_layout(pad=.8)
    output=a.output or a.input.parent/f"task_duration_{a.gpus}gpu.png"
    save_figure(fig,output)
    with output.with_suffix(".csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for r in rows: print(r)
    print(f"Saved {output.with_suffix('')}: PNG, SVG, PDF and CSV")

if __name__=="__main__":main()
