"""Poster versions of three figures, using saved summaries (no GPU execution)."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import statistics as stats
import textwrap
from matplotlib.patches import Patch
from plotting import plt, save_figure

BLUE="#2166ac"
ORANGE="#e69f00"
GRAY="#a9afb7"
GREEN="#238b45"

def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))

def frame(title, subtitle):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.set_title(title + "\n" + subtitle, fontsize=12, pad=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=11)
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    return fig, ax

def footer(fig, text):
    ax = fig.axes[0]
    y = -.20 if "Throughput" in ax.get_title() else -.36
    ax.text(.5, y, textwrap.fill(text, 112), transform=ax.transAxes,
            ha="center", va="top", fontsize=8)

def finish(fig, path):
    fig.tight_layout(pad=.8)
    save_figure(fig, path)

def paired_bars(ax, experts, counts, color, bottoms=None):
    artists=[]
    for ng,offset in [(1,-.19),(4,.19)]:
        vals=[counts[ng,e] for e in experts]
        artists.append(ax.bar([i+offset for i in range(len(experts))],vals,.33,
                             bottom=[bottoms[ng,e] for e in experts] if bottoms else None,
                             color=color,edgecolor="white",linewidth=.9,
                             hatch=None))
    return artists

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study",type=Path,default=Path("project/results/expert_granularity/study_1vs4_v2"))
    a=p.parse_args()
    base=a.study/"baseline";trace=a.study/"trace"
    plt.rcParams.update({"font.size":11,"axes.labelsize":12,"legend.fontsize":9})
    summary=[r for r in read(base/"summary.csv") if r["source"].startswith("Measured")]
    times={(int(r["ngpu"]),int(r["E"])):float(r["mean_ms"]) for r in summary}
    experts=sorted({e for ng,e in times})
    if any((ng,e) not in times for ng in (1,4) for e in experts):
        raise ValueError("Poster comparison requires both GPU counts for each expert count")
    fig,ax=frame("Input Throughput: 1 GPU vs. 4 GPUs",
                 "Weak scaling: 4,096 input tokens per GPU • BF16 gated MoE")
    for ng,color,marker in [(1,BLUE,"o"),(4,ORANGE,"s")]:
        values=[4096*ng/times[ng,e]/1000 for e in experts]
        ax.plot(range(len(experts)),values,color=color,marker=marker,markersize=6,
                linewidth=1.5,label=f"{ng} GPU" if ng==1 else "4 GPUs")
    for i,e in enumerate(experts):
        gain=4*times[1,e]/times[4,e]
        y=4096*4/times[4,e]/1000
        ax.annotate(f"{gain:.2f}× vs 1 GPU",(i,y),xytext=(0,22),textcoords="offset points",
                    ha="left" if i==0 else ("right" if i==len(experts)-1 else "center"),
                    fontsize=11,fontweight="normal")
    ax.set_ylim(0,max(4096*4/times[4,e]/1000 for e in experts)*1.22)
    ax.set_xticks(range(len(experts)),[str(e) for e in experts])
    ax.set_xlabel("Total number of experts")
    ax.set_ylabel("Aggregate input throughput\n(million tokens/s)")
    ax.legend(frameon=False,loc="upper right",ncol=2)
    footer(fig, "Throughput = total input tokens / mean layer latency (5 runs).")
    finish(fig,base/"throughput_vs_experts.png")

    rows=read(trace/"block_time_breakdown.csv")
    breakdown={(int(r["ngpu"]),int(r["E"])):r for r in rows}
    fig,ax=frame("Processor Block-Time Breakdown: 1 GPU vs. 4 GPUs",
                 "Post-warmup traces • task execution, mailbox waiting and other time")
    bottom={(ng,e):0. for ng in (1,4) for e in experts}
    for state,color in [("task",BLUE),("wait",ORANGE),("other",GRAY)]:
        vals={k:float(breakdown[k][state+"_mean_pct"]) for k in bottom}
        artists=paired_bars(ax,experts,vals,color,bottom)
        for ng,bars in zip((1,4),artists):
            for e,bar in zip(experts,bars):
                if state in ("task","wait"):
                    value=vals[ng,e]
                    ax.text(bar.get_x()+bar.get_width()/2,bottom[ng,e]+value/2,
                            f"{value:.1f}%",ha="center",va="center",
                            color="white" if state=="task" else "black",
                            fontsize=11 if state=="task" else 9.5)
        for k in bottom:bottom[k]+=vals[k]
    ax.set_xticks(range(len(experts)),[str(e) for e in experts])
    ax.set_xlabel("Total number of experts")
    ax.set_ylabel("Recorded processor block-time (%)")
    ax.set_ylim(0,102)
    for i in range(len(experts)):
        for ng,offset in [(1,-.19),(4,.19)]:
            ax.text(i+offset,-.025,str(ng)+" GPU"+("s" if ng==4 else ""),
                    transform=ax.get_xaxis_transform(),ha="center",va="top",fontsize=9)
    ax.tick_params(axis="x",pad=18)
    ax.legend(handles=[Patch(facecolor=BLUE,label="Task execution"),
                       Patch(facecolor=ORANGE,label="Mailbox waiting"),
                       Patch(facecolor=GRAY,label="Other")],
              loc="upper center",bbox_to_anchor=(.5,-.22),ncol=3,frameon=False)
    footer(fig, "Mean of 5 runs. OS block excluded. Task time includes transfers/signaling; waiting is not scheduler cost.")
    finish(fig,trace/"block_time_breakdown.png")

    samples=defaultdict(list)
    for r in read(trace/"task_metrics.csv"):
        samples[int(r["ngpu"]),int(r["E"]),r["task_type"]].append(float(r["count_per_gpu"]))
    fig,ax=frame("128 experts produce 33% more scheduled tasks",
                 "Compared with 16 / 64 experts • tasks per GPU per captured MoE layer")
    bottom={(ng,e):0. for ng in (1,4) for e in experts}
    for kind,color in [("0",BLUE),("1",GREEN),("2",ORANGE)]:
        vals={k:stats.mean(samples[k[0],k[1],kind]) for k in bottom}
        paired_bars(ax,experts,vals,color,bottom)
        for k in bottom:bottom[k]+=vals[k]
    for i,e in enumerate(experts):
        for ng,offset in [(1,-.19),(4,.19)]:
            ax.text(i+offset,bottom[ng,e]+170,f"{bottom[ng,e]:,.0f}",ha="center",fontsize=11)
    ax.set_ylim(0,max(bottom.values())*1.18)
    ax.set_xticks(range(len(experts)),[str(e) for e in experts])
    ax.set_xlabel("Total number of experts")
    ax.set_ylabel("Scheduled tasks per GPU per layer")
    for i in range(len(experts)):
        for ng,offset in [(1,-.19),(4,.19)]:
            ax.text(i+offset,-.025,str(ng)+" GPU"+("s" if ng==4 else ""),
                    transform=ax.get_xaxis_transform(),ha="center",va="top",fontsize=9)
    ax.tick_params(axis="x",pad=18)
    ax.legend(handles=[Patch(facecolor=BLUE,label="GEMM0: gated projection"),
                       Patch(facecolor=GREEN,label="GEMM1: output projection"),
                       Patch(facecolor=ORANGE,label="Output combine")],
              loc="upper center",bbox_to_anchor=(.5,-.22),ncol=3,frameon=False,fontsize=10)
    footer(fig, "Mean of 5 captures. Counts normalized per GPU; they do not measure useful FLOPs or scheduler overhead.")
    finish(fig,trace/"task_count_comparison.png")
    print("Updated three poster figures in PNG, SVG and PDF.")

if __name__=="__main__":main()
