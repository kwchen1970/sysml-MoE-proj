#!/usr/bin/env python3
"""Summarize interleaved A/B CSV (tag,round,S,H,I,E,k,latency_ms,error_pct) vs a baseline tag."""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def load_rows(path: Path):
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("tag,") or line == "DONE" or line.startswith("FAIL"):
                continue
            if "," not in line:
                continue
            p = [x.strip() for x in line.split(",")]
            if len(p) < 9 or p[0] in {"NVIDIA", "===", "SMOKE"}:
                continue
            try:
                rows.append(
                    {
                        "tag": p[0],
                        "round": int(p[1]),
                        "S": int(p[2]),
                        "H": int(p[3]),
                        "I": int(p[4]),
                        "E": int(p[5]),
                        "k": int(p[6]),
                        "latency_ms": float(p[7]),
                        "error_pct": float(p[8]),
                    }
                )
            except ValueError:
                continue
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", type=Path, help="A/B harness stdout or extracted CSV")
    ap.add_argument("--base", required=True, help="Baseline tag (e.g. HYB17)")
    ap.add_argument("--cand", required=True, help="Candidate tag (e.g. NCU1)")
    ap.add_argument("--anchor", default="4096,2048,1408,16,6", help="S,H,I,E,k for anchor line")
    args = ap.parse_args()

    rows = load_rows(args.csv)
    if not rows:
        print("no data rows", file=sys.stderr)
        return 1

    by = defaultdict(list)
    err = defaultdict(list)
    for r in rows:
        key = (r["tag"], r["S"], r["H"], r["I"], r["E"], r["k"])
        by[key].append(r["latency_ms"])
        err[key].append(r["error_pct"])

    aS, aH, aI, aE, ak = (int(x) for x in args.anchor.split(","))
    ratios = []
    print(f"shape, {args.base}_ms, {args.cand}_ms, ratio, d_pct, {args.cand}_err")
    shapes = sorted({k[1:] for k in by if k[0] in {args.base, args.cand}})
    for shape in shapes:
        b = by.get((args.base, *shape))
        c = by.get((args.cand, *shape))
        if not b or not c:
            print(f"{shape}, MISSING", file=sys.stderr)
            continue
        bm, cm = statistics.mean(b), statistics.mean(c)
        ratio = cm / bm
        ratios.append(ratio)
        em = statistics.mean(err[(args.cand, *shape)])
        mark = "  <-- anchor" if shape == (aS, aH, aI, aE, ak) else ""
        print(
            f"{shape[0]},{shape[1]},{shape[2]},{shape[3]},{shape[4]}, "
            f"{bm:.4f}, {cm:.4f}, {ratio:.4f}, {(ratio-1)*100:+.2f}%, {em:.4f}{mark}"
        )

    if not ratios:
        return 2
    geo = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
    print(f"geo_mean_ratio={geo:.4f} ({(geo-1)*100:+.2f}% vs {args.base}, n={len(ratios)})")
    keep = geo < 1.0 and all(
        statistics.mean(err[(args.cand, *s)]) < 1.0 for s in shapes if (args.cand, *s) in err
    )
    print(f"keep_candidate={'YES' if keep and geo < 0.995 else 'MAYBE' if keep else 'NO'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
