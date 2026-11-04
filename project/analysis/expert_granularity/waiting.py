"""Compute waiting over explicitly declared complete processor observation windows."""
import argparse,csv,statistics
from collections import defaultdict
from pathlib import Path
from intervals import load
from plotting import lineplot

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("events",type=Path); p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(); groups=load(a.events); totals=defaultdict(lambda:[0.,0.])
    for key,rows in groups.items():
        if key[7]!="processor": continue
        # Accumulate block-time within an invocation; do not pool ranks as repetitions.
        t=totals[key[:4]]
        t[0]+=sum(r["end_ns"]-r["start_ns"] for r in rows if r["state"]=="wait")
        t[1]+=rows[0]["window_end_ns"]-rows[0]["window_start_ns"]
    if not totals: p.error("No processor intervals")
    a.output.mkdir(parents=True,exist_ok=True); rows=[]; points=defaultdict(list)
    for (run,inv,ng,e),(wait,total) in sorted(totals.items()):
        rows.append(dict(run_id=run,invocation=inv,ngpu=ng,E=e,wait_block_ns=wait,observed_block_ns=total,waiting_fraction=wait/total))
        points[ng,e].append(100*wait/total)
    with (a.output/"waiting.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    plots=defaultdict(list)
    for (ng,e),values in sorted(points.items()):
        plots[f"{ng} GPU"].append((e,statistics.mean(values),statistics.stdev(values) if len(values)>1 else 0))
    lineplot(a.output/"waiting_vs_experts.svg",plots,"Recorded processor waiting; attribution requires dependency evidence","Waiting (% of recorded processor block-time)")
    rank_lanes=defaultdict(list); rank_tasks=defaultdict(list)
    for key,events in groups.items():
        rank_lanes[key[:5]].extend(events)
        if key[7]=="processor": rank_tasks[key[:5]].append(events)
    invstats=defaultdict(list); taskstats=defaultdict(list)
    for key,lanes in rank_tasks.items():
        durations=[e["end_ns"]-e["start_ns"] for lane in lanes for e in lane if e["state"]=="task"]
        actors=rank_lanes[key]
        span=max(e["window_end_ns"] for e in actors)-min(e["window_start_ns"] for e in actors)
        bound=max(sum(durations)/len(lanes),max(durations,default=0))
        invstats[key[:4]].append((span/1e6,bound/1e6))
        for lane in lanes:
            for e in lane:
                if e["state"]=="task":
                    taskstats[key[:4]+(e.get("task_type","unknown"),)].append((e["end_ns"]-e["start_ns"])/1000)
    workrows=[]; workplots=defaultdict(list); byconfig=defaultdict(list)
    for (run,inv,ng,e),ranks in sorted(invstats.items()):
        span=max(x[0] for x in ranks);bound=max(x[1] for x in ranks)
        workrows.append(dict(run_id=run,invocation=inv,ngpu=ng,E=e,captured_actor_span_ms=span,task_work_bound_ms=bound,gap_ms=span-bound))
        byconfig[ng,e].append((span,bound))
    for (ng,e),values in sorted(byconfig.items()):
        for col,label in [(0,"captured actor span"),(1,"task work bound")]:
            vals=[v[col] for v in values]
            workplots[f"{ng} GPU {label}"].append((e,statistics.mean(vals),statistics.stdev(vals) if len(vals)>1 else 0))
    with (a.output/"work_bound.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(workrows[0]));w.writeheader();w.writerows(workrows)
    lineplot(a.output/"span_vs_work_bound.svg",workplots,"Captured actor span vs task-only work bound; gap is not scheduler cost","Time (ms)")
    taskrows=[]
    for key,values in sorted(taskstats.items()):
        run,inv,ng,e,kind=key
        taskrows.append(dict(run_id=run,invocation=inv,ngpu=ng,E=e,task_type=kind,count=len(values),mean_us=statistics.mean(values),min_us=min(values),max_us=max(values)))
    if taskrows:
        with (a.output/"task_durations.csv").open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(taskrows[0]));w.writeheader();w.writerows(taskrows)
    print("Recorded processor block-time only. Spans are per rank; GPU clocks are never aligned.")
if __name__=="__main__": main()
