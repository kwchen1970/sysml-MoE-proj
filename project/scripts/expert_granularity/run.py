"""Run baseline or device-trace experiments; retain explicit invocation metadata."""
import argparse, csv, datetime, hashlib, json, os, random, subprocess
from pathlib import Path

FIELDS = "rank dtype S H I E k time_ms error_pct EC GPU SM Topology MLPType bM bN0 bK0 bN1 bK1 threads blocks_per_SM SMs blocks rtol atol graph_launches warmup runs workspace_MiB".split()

def parse_output(text, ngpu, expected):
    rows = []
    for line in text.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if not parts or not parts[0].isdigit():
            continue
        if len(parts) != len(FIELDS):
            raise ValueError(f"Malformed benchmark row: {line}")
        row = dict(zip(FIELDS, parts))
        if tuple(int(row[k]) for k in ("S", "H", "I", "E", "k")) != expected:
            raise ValueError("Benchmark printed an unexpected configuration")
        if not (float(row["time_ms"]) > 0 and 0 <= float(row["error_pct"]) <= 100):
            raise ValueError("Invalid timing/correctness result")
        rows.append(row)
    if sorted(int(r["rank"]) for r in rows) != list(range(ngpu)):
        raise ValueError(f"Expected exactly ranks 0..{ngpu-1}; got {[r['rank'] for r in rows]}")
    return rows

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", type=Path, default=Path("csrc/cmake-build-release/testFlashMoE"))
    p.add_argument("--gpus", nargs="+", type=int, choices=[1, 4], default=[1])
    p.add_argument("--experts", nargs="+", type=int, choices=[16, 32, 64, 128], default=[16, 64, 128])
    p.add_argument("--repetitions", type=int, default=5)
    p.add_argument("--seed", type=int, default=36, help="Configuration order only; benchmark data seeds remain random")
    p.add_argument("--output", type=Path)
    p.add_argument("--device-trace", action="store_true", help="Capture one launch per invocation; requires trace build")
    a = p.parse_args()
    if a.repetitions < 2: p.error("Use at least two repetitions")
    binary = a.binary.resolve()
    if not binary.is_file(): p.error(f"Build the benchmark first: {binary}")
    if not os.environ.get("SLURM_JOB_ID"): p.error("Run inside a Slurm GPU allocation")
    run_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + os.environ["SLURM_JOB_ID"]
    out = a.output or Path(os.environ["SCRATCH"]) / "moe/expert_granularity" / run_id
    out.mkdir(parents=True, exist_ok=False)
    def git(*args): return subprocess.check_output(["git", "--no-optional-locks", *args], text=True).strip()
    meta = dict(run_id=run_id, revision=git("rev-parse", "HEAD"), diff=git("diff", "--", "csrc", "flashmoe"),
                binary=str(binary), binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                job_id=os.environ["SLURM_JOB_ID"], order_seed=a.seed, routing_seeds="uncontrolled benchmark random_device",
                timing_scope="dispatch/expert/combine; excludes gate",
                trace_scope="one post-warmup launch after timed repeats" if a.device_trace else "disabled",
                timing_note="trace-build timed repeats have recorder disabled" if a.device_trace else "baseline", arguments=vars(a).copy(),
                environment={k: os.environ.get(k) for k in ("MATHDX_ROOT", "CUDA_HOME", "NVSHMEM_HOME", "NVSHMEM_REMOTE_TRANSPORT", "LOADEDMODULES")})
    source_paths = ["csrc/CMakeLists.txt", "csrc/tests/flashmoe.cu", "csrc/tests/project_trace_host.cuh", "csrc/tests/single_node_init.cuh",
                    "project/scripts/expert_granularity/launch_single_node.sh",
                    "csrc/include/flashmoe/infra/project_trace.cuh", "csrc/include/flashmoe/processor.cuh",
                    "csrc/include/flashmoe/moe.cuh", "project/scripts/expert_granularity/run.py"]
    meta["source_sha256"] = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                              for p in source_paths if Path(p).is_file()}
    cache = binary.parent / "CMakeCache.txt"
    if cache.is_file():
        meta["build_settings"] = [line for line in cache.read_text().splitlines()
            if line.startswith(("CMAKE_BUILD_TYPE:", "CMAKE_CUDA_COMPILER:", "CMAKE_CUDA_FLAGS:",
                                "CMAKE_CUDA_ARCHITECTURES:", "FLASHMOE_PROJECT_TRACE:"))]
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    jobs = [(rep, ng, e) for rep in range(a.repetitions) for ng in sorted(set(a.gpus)) for e in sorted(set(a.experts))]
    random.Random(a.seed).shuffle(jobs)
    with (out / "measurements.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run_id", "invocation", "repetition", "ngpu"] + FIELDS); w.writeheader()
        for inv, (rep, ng, e) in enumerate(jobs):
            cmd = ["bash", str(Path(__file__).resolve().with_name("launch_single_node.sh")), str(ng), str(binary), "4096", "2048", "1408", str(e), "6", "8", "128", "128"]
            (out / f"{inv:03d}.command.json").write_text(json.dumps(cmd))
            print(f"invocation={inv} repetition={rep} GPUs={ng} experts={e}", flush=True)
            env = os.environ.copy()
            # Never inherit a capture path into an ordinary baseline run.
            env.pop("FLASHMOE_TRACE_PREFIX", None)
            if a.device_trace:
                env.update(FLASHMOE_TRACE_PREFIX=str(out.resolve()/f"events-{inv:03d}"),
                           FLASHMOE_TRACE_RUN_ID=run_id, FLASHMOE_TRACE_INVOCATION=str(inv))
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
            (out / f"{inv:03d}.stdout.log").write_text(result.stdout)
            (out / f"{inv:03d}.stderr.log").write_text(result.stderr)
            result.check_returncode()
            for row in parse_output(result.stdout, ng, (4096, 2048, 1408, e, 6)):
                w.writerow(dict(run_id=run_id, invocation=inv, repetition=rep, ngpu=ng, **row))
            f.flush()
    if a.device_trace:
        event_rows=[]
        with (out/"measurements.csv").open() as f:
            measurements=list(csv.DictReader(f))
        for inv, (rep, ng, e) in enumerate(jobs):
            for rank in range(ng):
                path=out/f"events-{inv:03d}.rank{rank}.csv"
                with path.open() as f:
                    rows=list(csv.DictReader(f))
                expected=[m for m in measurements
                          if int(m["invocation"])==inv and int(m["rank"])==rank]
                blocks=int(expected[0]["blocks"])
                if set(int(x["block_id"]) for x in rows)!=set(range(blocks)):
                    raise ValueError(f"Incomplete block coverage in {path}")
                if any(int(x["dropped_events"]) or x["run_id"]!=run_id or int(x["invocation"])!=inv
                       or int(x["rank"])!=rank or int(x["E"])!=e or int(x["ngpu"])!=ng for x in rows):
                    raise ValueError(f"Invalid trace metadata/overflow in {path}")
                event_rows.extend(rows)
        with (out/"events.csv").open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(event_rows[0])); w.writeheader();w.writerows(event_rows)
    print(f"Results: {out}", flush=True)

if __name__ == "__main__": main()
