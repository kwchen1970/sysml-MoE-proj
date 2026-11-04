import sys, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

log = sys.argv[1] if len(sys.argv) > 1 else "project/results/raw/sweep_interactive.log"
lines = [l for l in open(log) if l[:1].isdigit()]
cols = [c.strip() for c in "rank, dtype, S, H, I, E, k, time_ms, error_pct, EC, GPU, SM, Topology, MLPType, bM, bN0, bK0, bN1, bK1, threads, blocks_per_SM, SMs, blocks, rtol, atol, graph_launches, warmup, runs, workspace_MiB".split(",")]
df = pd.DataFrame([[x.strip() for x in l.split(",")] for l in lines], columns=cols)
for c in ["rank","S","H","I","E","k","time_ms","error_pct","EC","blocks","workspace_MiB"]:
    df[c] = pd.to_numeric(df[c])
# script runs 1-GPU section first (12 configs x 3 reps = 36 rows), then 4-GPU (144 rows)
df["ngpu"] = [1]*36 + [4]*(len(df)-36)
df.to_csv("project/results/sweep_1node_all.csv", index=False)

keys = ["ngpu","S","H","I","E","k"]
summ = (df.groupby(keys)
          .agg(time_ms_mean=("time_ms","mean"), time_ms_std=("time_ms","std"),
               max_error_pct=("error_pct","max"), EC=("EC","first"),
               blocks=("blocks","first"), workspace_MiB=("workspace_MiB","first"))
          .reset_index())
summ.to_csv("project/results/sweep_1node_summary.csv", index=False)
print(summ.to_string(index=False))

def plot(sub, x, title, fname, logx=False):
    plt.figure(figsize=(5,3.5))
    for ng, g in sub.groupby("ngpu"):
        g = g.sort_values(x)
        plt.errorbar(g[x], g.time_ms_mean, yerr=g.time_ms_std, marker="o", capsize=3, label=f"{ng} GPU")
    if logx: plt.xscale("log", base=2)
    plt.xlabel(x); plt.ylabel("latency (ms)"); plt.title(title); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"project/results/plots/{fname}", dpi=150); plt.close()

plot(summ[(summ.H==2048)&(summ.I==1408)&(summ.E==64)&(summ.k==6)], "S",
     "Latency vs tokens/GPU (E=64, I=1408, k=6)", "latency_vs_tokens.png", logx=True)
plot(summ[(summ.S==4096)&(summ.I==1408)&(summ.k==6)], "E",
     "Latency vs #experts (S=4096, I=1408, k=6)", "latency_vs_experts.png", logx=True)
plot(summ[(summ.S==4096)&(summ.E==16)&(summ.k==2)], "I",
     "Latency vs expert FFN dim (S=4096, E=16, k=2)", "latency_vs_ffn.png")
print("\nmax error across all runs (%):", df.error_pct.max())
