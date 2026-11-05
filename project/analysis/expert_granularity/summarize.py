"""Validate new results and compare with Jo's baseline; requires only Python 3."""
import argparse, csv, math, statistics
from collections import defaultdict
from pathlib import Path
from plotting import lineplot
CONFIG=("ngpu","S","H","I","E","k")
def summarize(path, legacy=False):
    groups=defaultdict(list); maximum=0
    with path.open() as f:
        input_rows=list(csv.DictReader(f))
    for r in input_rows:
        if (int(r["S"]),int(r["H"]),int(r["I"]),int(r["k"])) != (4096,2048,1408,6): continue
        key=tuple(int(r[k]) for k in CONFIG)
        t=float(r["time_ms"]); err=float(r["error_pct"])
        if not math.isfinite(t) or t<=0 or not math.isfinite(err) or not 0<=err<=100: raise ValueError("Invalid measurement")
        maximum=max(maximum,err)
        if legacy: groups[key].append(t)
        else: groups[key].append((r["run_id"],r["invocation"],int(r["rank"]),t))
    out=[]
    for key,records in sorted(groups.items()):
        if legacy: values=records
        else:
            inv=defaultdict(list)
            for run,num,rank,t in records: inv[run,num].append((rank,t))
            for ranks in inv.values():
                if sorted(rank for rank,t in ranks)!=list(range(key[0])): raise ValueError(f"Missing/duplicate ranks: {key}")
            values=[max(t for rank,t in ranks) for ranks in inv.values()]
        out.append(dict(zip(CONFIG,key),mean_ms=statistics.mean(values),std_ms=statistics.stdev(values) if len(values)>1 else 0,n=len(values)))
    if not out: raise ValueError(f"No matching granularity rows in {path}")
    return out,maximum

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("measurements",type=Path,nargs="?")
    p.add_argument("--jo",type=Path,default=Path(__file__).resolve().parents[2]/"results/sweep_1node_all.csv")
    p.add_argument("--no-reference",action="store_true",help="Plot only the new measurements")
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    sources=[]
    if a.jo and not a.no_reference:
        sources.append(("Reference baseline (pooled ranks/repeats)", *summarize(a.jo, True)))

    if a.measurements:
        sources.append(("Measured (max rank per invocation)", *summarize(a.measurements)))
    if not sources: p.error("Provide measurements or enable the reference")
    allrows=[]; groups={}
    for name,rows,error in sources:
        print(f"{name}: max output-element mismatch = {error:.6f}% (nonzero requires investigation)")
        for r in rows:
            allrows.append(dict(source=name,**r))
            groups.setdefault(f"{name}, {r['ngpu']} GPU",[]).append((r["E"],r["mean_ms"],r["std_ms"]))
    with (a.output/"summary.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(allrows[0])); w.writeheader(); w.writerows(allrows)
    lineplot(a.output/"latency_vs_experts.svg",groups,"Expert granularity: S/GPU=4096, H=2048, I=1408, top-k=6","Dispatch–expert–combine latency (ms)")
    throughput={label:[(e,4096*int(label.split(', ')[-1].split()[0])/t/1000,0) for e,t,s in points] for label,points in groups.items()}
    lineplot(a.output/"throughput_vs_experts.svg",throughput,"Weak scaling; routing excluded; nominal input throughput","Million input tokens / second")
    print(f"Wrote summaries and Matplotlib PNG/SVG/PDF figures to {a.output}")
if __name__=="__main__": main()
