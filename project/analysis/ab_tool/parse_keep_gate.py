#!/usr/bin/env python3
"""Parse interleaved A/B job output and apply the FlashMoE keep gate.

KEEP only if:
  1. correctness: every successful row has error_pct <= error_max
  2. speed: geo_mean(cand/base) < 1.0  (cand faster than base; default base=HYB17)
  3. coverage: every shape has means for both cand and base

Also reports geo vs HYB2 / BK64 when those tags are present, plus anchor delta,
per-shape CV (noise), and FAIL line counts.

Examples:
  python parse_keep_gate.py project/results/shared_a_ab/hyb17-61377792.out \\
      --cand HYB17 --base HYB2 --out /tmp/hyb17_gate
  python parse_keep_gate.py /pscratch/.../ab/tool/noise/tool-noise-123.out \\
      --cand HYB17B --base HYB17A --identity
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ANCHOR = (4096, 2048, 1408, 16, 6)
ROW_PREFIX_OK = ("BK64,", "HYB", "SCHED,", "CAND,", "EXP,", "A,", "B,")


# Hard default: require ≥1% geo win so keeps clear the ~±1% identity floor
# measured on hyb17-61377792 (odd/even bootstrap). Raise further with
# --auto-margin or after the dedicated noise study.
DEFAULT_MIN_GEO_MARGIN_PCT = 1.0


def geo_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return math.exp(sum(math.log(v) for v in values) / len(values))


def geo_delta_pct(values: list[float]) -> float:
    g = geo_mean(values)
    return 100.0 * (g - 1.0) if g == g else float("nan")


def parse_rows(
    text: str,
    *,
    phase_substr: str | None = None,
) -> tuple[list[dict[str, Any]], list[str], dict[str, str]]:
    rows: list[dict[str, Any]] = []
    fails: list[str] = []
    meta: dict[str, str] = {}
    # Optional: restrict to one "==== PHASE ... ====" section (noise study).
    active = phase_substr is None
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("==== PHASE"):
            active = phase_substr is None or (phase_substr in line)
            continue
        if line.startswith("==== ") and phase_substr is not None:
            # Closing/other banner ends the active phase.
            if active and not line.startswith("==== PHASE"):
                active = False
            continue
        if not active:
            continue
        if line.startswith("META "):
            # META key=value key=value ...
            body = line[5:]
            for tok in body.split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    meta[k] = v
            continue
        if line.startswith("FAIL"):
            fails.append(line)
            continue
        if line.startswith("tag,round,S,H,I,E,k"):
            continue
        if "," not in line:
            continue
        # Accept any tag that produced the harness CSV schema.
        parts = line.split(",")
        if len(parts) < 9:
            continue
        tag = parts[0]
        # Skip smoke marker rounds written as round=smoke
        if parts[1] == "smoke":
            # Still record for correctness if schema matches.
            try:
                rows.append(
                    {
                        "tag": tag,
                        "round": -1,
                        "S": int(parts[2]),
                        "H": int(parts[3]),
                        "I": int(parts[4]),
                        "E": int(parts[5]),
                        "k": int(parts[6]),
                        "latency_ms": float(parts[7]),
                        "error_pct": float(parts[8]),
                        "smoke": True,
                    }
                )
            except ValueError:
                pass
            continue
        try:
            rows.append(
                {
                    "tag": tag,
                    "round": int(parts[1]),
                    "S": int(parts[2]),
                    "H": int(parts[3]),
                    "I": int(parts[4]),
                    "E": int(parts[5]),
                    "k": int(parts[6]),
                    "latency_ms": float(parts[7]),
                    "error_pct": float(parts[8]),
                    "smoke": False,
                }
            )
        except ValueError:
            continue
    return rows, fails, meta


def shape_key(row: dict[str, Any]) -> tuple[int, int, int, int, int]:
    return (row["S"], row["H"], row["I"], row["E"], row["k"])


def summarize(rows: list[dict[str, Any]]) -> dict[tuple, dict[str, Any]]:
    buckets: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("smoke"):
            continue
        key = (row["tag"],) + shape_key(row)
        buckets[key].append(row)
    out: dict[tuple, dict[str, Any]] = {}
    for key, vals in buckets.items():
        # Preserve round order for odd/even bootstrap.
        vals = sorted(vals, key=lambda v: v["round"])
        lats = [v["latency_ms"] for v in vals]
        errs = [v["error_pct"] for v in vals]
        mean = statistics.mean(lats)
        out[key] = {
            "mean": mean,
            "median": statistics.median(lats),
            "lo": min(lats),
            "hi": max(lats),
            "n": len(lats),
            "stdev": statistics.stdev(lats) if len(lats) > 1 else 0.0,
            "cv_pct": (100.0 * statistics.stdev(lats) / mean) if len(lats) > 1 and mean else 0.0,
            "error_max": max(errs) if errs else float("nan"),
            "error_mean": statistics.mean(errs) if errs else float("nan"),
            "lats": lats,
        }
    return out


def odd_even_identity_geo(stats: dict[tuple, dict[str, Any]], tag: str) -> dict[str, Any]:
    """Estimate identity noise by splitting one tag's rounds into odd/even means."""
    ratios = []
    shapes = []
    for key, st in stats.items():
        if key[0] != tag or st["n"] < 4:
            continue
        lats = st["lats"]
        a = statistics.mean(lats[0::2])
        b = statistics.mean(lats[1::2])
        if a > 0 and b > 0:
            ratios.append(a / b)
            shapes.append(key[1:])
    if not ratios:
        return {"geo_delta_pct": float("nan"), "n_shapes": 0, "tag": tag}
    geo = geo_mean(ratios)
    return {
        "tag": tag,
        "geo_delta_pct": 100.0 * (geo - 1.0),
        "abs_geo_delta_pct": abs(100.0 * (geo - 1.0)),
        "n_shapes": len(ratios),
        "recommended_min_geo_margin_pct": max(1.0, abs(100.0 * (geo - 1.0)) * 1.5),
    }


def compare(
    stats: dict[tuple, dict[str, Any]],
    cand: str,
    base: str,
) -> dict[str, Any]:
    cand_shapes = {k[1:]: v for k, v in stats.items() if k[0] == cand}
    base_shapes = {k[1:]: v for k, v in stats.items() if k[0] == base}
    common = sorted(set(cand_shapes) & set(base_shapes))
    ratios = []
    per_shape = []
    for shape in common:
        c = cand_shapes[shape]
        b = base_shapes[shape]
        ratio = c["mean"] / b["mean"] if b["mean"] else float("nan")
        delta_pct = 100.0 * (ratio - 1.0)
        ratios.append(ratio)
        per_shape.append(
            {
                "S": shape[0],
                "H": shape[1],
                "I": shape[2],
                "E": shape[3],
                "k": shape[4],
                "cand_ms": c["mean"],
                "base_ms": b["mean"],
                "ratio": ratio,
                "delta_pct": delta_pct,
                "cand_cv_pct": c["cv_pct"],
                "base_cv_pct": b["cv_pct"],
                "cand_error_max": c["error_max"],
                "base_error_max": b["error_max"],
                "n_cand": c["n"],
                "n_base": b["n"],
            }
        )
    geo = geo_mean(ratios) if ratios else float("nan")
    deltas = [100.0 * (r - 1.0) for r in ratios]
    # Leave-one-out geo deltas ≈ cheap CI without numpy.
    loo = []
    if len(ratios) >= 3:
        for i in range(len(ratios)):
            loo.append(geo_delta_pct(ratios[:i] + ratios[i + 1 :]))
    anchor = next((p for p in per_shape if (p["S"], p["H"], p["I"], p["E"], p["k"]) == ANCHOR), None)
    return {
        "cand": cand,
        "base": base,
        "n_shapes": len(per_shape),
        "geo_ratio": geo,
        "geo_delta_pct": 100.0 * (geo - 1.0) if geo == geo else float("nan"),
        "median_delta_pct": statistics.median(deltas) if deltas else float("nan"),
        "geo_loo_lo_pct": min(loo) if loo else float("nan"),
        "geo_loo_hi_pct": max(loo) if loo else float("nan"),
        "wins": sum(1 for p in per_shape if p["delta_pct"] < -1e-9),
        "losses": sum(1 for p in per_shape if p["delta_pct"] > 1e-9),
        "ties": sum(1 for p in per_shape if abs(p["delta_pct"]) <= 1e-9),
        "anchor": anchor,
        "per_shape": per_shape,
        "missing_cand": sorted(set(base_shapes) - set(cand_shapes)),
        "missing_base": sorted(set(cand_shapes) - set(base_shapes)),
    }


def keep_decision(
    cmp: dict[str, Any],
    rows: list[dict[str, Any]],
    fails: list[str],
    *,
    cand: str,
    error_max: float,
    min_geo_margin_pct: float,
    require_anchor_win: bool,
) -> dict[str, Any]:
    cand_rows = [r for r in rows if r["tag"] == cand and not r.get("smoke")]
    err_viol = [r for r in cand_rows if r["error_pct"] > error_max]
    reasons = []
    ok_correct = not err_viol and not fails
    if fails:
        reasons.append(f"{len(fails)} FAIL line(s)")
    if err_viol:
        reasons.append(f"{len(err_viol)} cand rows exceed error_max={error_max}")
    if cmp["n_shapes"] == 0:
        reasons.append("no overlapping shapes")
        ok_correct = False
    if cmp["missing_cand"] or cmp["missing_base"]:
        reasons.append("incomplete shape coverage")
        ok_correct = False

    geo = cmp["geo_ratio"]
    ok_fast = geo == geo and geo < (1.0 - min_geo_margin_pct / 100.0)
    if not (geo == geo):
        reasons.append("geo undefined")
    elif not ok_fast:
        reasons.append(
            f"geo {cmp['geo_delta_pct']:+.3f}% not faster than margin {min_geo_margin_pct:g}%"
        )

    ok_anchor = True
    if require_anchor_win:
        if cmp["anchor"] is None:
            ok_anchor = False
            reasons.append("anchor shape missing")
        elif cmp["anchor"]["delta_pct"] >= 0:
            ok_anchor = False
            reasons.append(f"anchor {cmp['anchor']['delta_pct']:+.3f}% not a win")

    # Measurement-quality hint from cand CVs (not a hard gate unless margin set).
    cand_cvs = [p["cand_cv_pct"] for p in cmp["per_shape"] if p["n_cand"] > 1]
    mean_cv = statistics.mean(cand_cvs) if cand_cvs else float("nan")
    bootstrap = cmp.get("bootstrap_identity") or {}
    boot_abs = bootstrap.get("abs_geo_delta_pct")
    if mean_cv == mean_cv and abs(cmp["geo_delta_pct"]) < mean_cv and ok_fast:
        reasons.append(
            f"WARNING geo |{cmp['geo_delta_pct']:.3f}%| < mean cand CV {mean_cv:.2f}% "
            "(near noise; raise ROUNDS or --min-geo-margin-pct)"
        )
    if (
        boot_abs == boot_abs
        and boot_abs is not None
        and abs(cmp["geo_delta_pct"]) < boot_abs
        and ok_fast
    ):
        reasons.append(
            f"WARNING geo |{cmp['geo_delta_pct']:.3f}%| < odd/even identity "
            f"|{boot_abs:.3f}%| on {cand}"
        )
    loo_hi = cmp.get("geo_loo_hi_pct")
    if (
        loo_hi == loo_hi
        and loo_hi is not None
        and ok_fast
        and loo_hi > -min_geo_margin_pct
    ):
        reasons.append(
            f"WARNING leave-one-out geo hi {loo_hi:+.3f}% fails margin "
            f"{min_geo_margin_pct:g}% (shape-fragile keep)"
        )

    keep = bool(ok_correct and ok_fast and ok_anchor)
    return {
        "keep": keep,
        "verdict": "KEEP" if keep else "NO_KEEP",
        "ok_correct": ok_correct,
        "ok_fast": ok_fast,
        "ok_anchor": ok_anchor,
        "reasons": reasons,
        "error_violations": len(err_viol),
        "fail_lines": len(fails),
        "mean_cand_cv_pct": mean_cv,
        "bootstrap_identity": bootstrap,
    }


def write_outputs(
    out_dir: Path,
    stats: dict[tuple, dict[str, Any]],
    cmp: dict[str, Any],
    decision: dict[str, Any],
    extras: list[dict[str, Any]],
    meta: dict[str, str],
    tags: list[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Wide per-shape means CSV (cand/base + optional peers).
    shapes = sorted({k[1:] for k in stats})
    fieldnames = ["S", "H", "I", "E", "k"] + tags
    # deltas for primary compare
    fieldnames += [f"d_{cmp['cand']}_vs_{cmp['base']}_pct"]
    with (out_dir / "compare.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for shape in shapes:
            row: dict[str, Any] = {
                "S": shape[0],
                "H": shape[1],
                "I": shape[2],
                "E": shape[3],
                "k": shape[4],
            }
            for tag in tags:
                st = stats.get((tag,) + shape)
                row[tag] = f"{st['mean']:.6f}" if st else ""
            c = stats.get((cmp["cand"],) + shape)
            b = stats.get((cmp["base"],) + shape)
            if c and b and b["mean"]:
                row[f"d_{cmp['cand']}_vs_{cmp['base']}_pct"] = f"{100.0 * (c['mean']/b['mean'] - 1.0):.3f}"
            else:
                row[f"d_{cmp['cand']}_vs_{cmp['base']}_pct"] = ""
            writer.writerow(row)

    # Long raw means with CV for noise analysis.
    with (out_dir / "shape_stats.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "tag",
                "S",
                "H",
                "I",
                "E",
                "k",
                "mean_ms",
                "median_ms",
                "lo",
                "hi",
                "n",
                "stdev",
                "cv_pct",
                "error_max",
            ],
        )
        writer.writeheader()
        for (tag, S, H, I, E, k), st in sorted(stats.items()):
            writer.writerow(
                {
                    "tag": tag,
                    "S": S,
                    "H": H,
                    "I": I,
                    "E": E,
                    "k": k,
                    "mean_ms": f"{st['mean']:.6f}",
                    "median_ms": f"{st['median']:.6f}",
                    "lo": f"{st['lo']:.6f}",
                    "hi": f"{st['hi']:.6f}",
                    "n": st["n"],
                    "stdev": f"{st['stdev']:.6f}",
                    "cv_pct": f"{st['cv_pct']:.3f}",
                    "error_max": f"{st['error_max']:.6f}",
                }
            )

    report = {
        "meta": meta,
        "primary": {
            "geo_ratio": cmp["geo_ratio"],
            "geo_delta_pct": cmp["geo_delta_pct"],
            "wins": cmp["wins"],
            "losses": cmp["losses"],
            "n_shapes": cmp["n_shapes"],
            "anchor": cmp["anchor"],
            "cand": cmp["cand"],
            "base": cmp["base"],
        },
        "extras": [
            {
                "cand": e["cand"],
                "base": e["base"],
                "geo_delta_pct": e["geo_delta_pct"],
                "wins": e["wins"],
                "losses": e["losses"],
                "anchor": e["anchor"],
            }
            for e in extras
        ],
        "decision": decision,
        "tags": tags,
    }
    (out_dir / "keep_gate.json").write_text(json.dumps(report, indent=2, default=str) + "\n")

    lines = [
        f"VERDICT {decision['verdict']}",
        f"primary {cmp['cand']} vs {cmp['base']}: geo {cmp['geo_delta_pct']:+.3f}% "
        f"(median {cmp.get('median_delta_pct', float('nan')):+.3f}%) "
        f"({cmp['wins']}W/{cmp['losses']}L/{cmp.get('ties', 0)}T over {cmp['n_shapes']} shapes)",
    ]
    loo_lo = cmp.get("geo_loo_lo_pct")
    loo_hi = cmp.get("geo_loo_hi_pct")
    if loo_lo == loo_lo and loo_hi == loo_hi:
        lines.append(f"geo leave-one-out range: [{loo_lo:+.3f}%, {loo_hi:+.3f}%]")
    if cmp["anchor"]:
        a = cmp["anchor"]
        lines.append(
            f"anchor: {a['cand_ms']:.4f} vs {a['base_ms']:.4f} ms ({a['delta_pct']:+.3f}%)"
        )
    mean_cv = decision.get("mean_cand_cv_pct")
    if mean_cv is not None and mean_cv == mean_cv:
        lines.append(f"measurement: mean cand CV {mean_cv:.2f}% across shapes")
    boot = cmp.get("bootstrap_identity") or decision.get("bootstrap_identity") or {}
    if boot.get("n_shapes"):
        lines.append(
            f"bootstrap identity ({boot.get('tag')} odd/even): "
            f"geo {boot['geo_delta_pct']:+.3f}% over {boot['n_shapes']} shapes; "
            f"recommend --min-geo-margin-pct {boot['recommended_min_geo_margin_pct']:.2f}"
        )
    for e in extras:
        lines.append(f"extra {e['cand']} vs {e['base']}: geo {e['geo_delta_pct']:+.3f}%")
    if decision["reasons"]:
        lines.append("reasons: " + "; ".join(decision["reasons"]))
    (out_dir / "keep_gate.txt").write_text("\n".join(lines) + "\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("job_out", type=Path, help="A/B job .out file (or raw CSV with header)")
    p.add_argument("--cand", required=True, help="Candidate tag (must beat --base)")
    p.add_argument("--base", default="HYB17", help="Baseline tag to beat (default HYB17)")
    p.add_argument("--also-vs", nargs="*", default=["HYB2", "BK64"], help="Extra baselines to report")
    p.add_argument("--error-max", type=float, default=0.25, help="Max allowed error_pct for cand")
    p.add_argument(
        "--min-geo-margin-pct",
        type=float,
        default=DEFAULT_MIN_GEO_MARGIN_PCT,
        help=f"Require geo delta_pct <= -margin (default {DEFAULT_MIN_GEO_MARGIN_PCT})",
    )
    p.add_argument(
        "--auto-margin",
        action="store_true",
        help="Raise margin to max(default, odd/even bootstrap×1.5 recommendation)",
    )
    p.add_argument(
        "--require-anchor-win",
        action="store_true",
        help="Also require anchor shape delta < 0",
    )
    p.add_argument(
        "--identity",
        action="store_true",
        help="Noise mode: print |geo-1| and mean CV; never KEEP",
    )
    p.add_argument(
        "--phase",
        default=None,
        help="Only parse rows after an '==== PHASE <substr>' banner (noise study)",
    )
    p.add_argument("--out", type=Path, required=True, help="Output directory for CSV/JSON/txt")
    args = p.parse_args()

    text = args.job_out.read_text(errors="replace")
    rows, fails, meta = parse_rows(text, phase_substr=args.phase)
    if not rows:
        print(f"no data rows parsed from {args.job_out}", file=sys.stderr)
        return 2

    stats = summarize(rows)
    tags = sorted({r["tag"] for r in rows})
    if args.cand not in tags:
        print(f"cand tag {args.cand!r} not in {tags}", file=sys.stderr)
        return 3
    if args.base not in tags:
        print(f"base tag {args.base!r} not in {tags}", file=sys.stderr)
        return 3

    primary = compare(stats, args.cand, args.base)
    boot = odd_even_identity_geo(stats, args.cand)
    primary["bootstrap_identity"] = boot
    margin = args.min_geo_margin_pct
    if args.auto_margin and boot.get("recommended_min_geo_margin_pct") == boot.get(
        "recommended_min_geo_margin_pct"
    ):
        margin = max(margin, float(boot["recommended_min_geo_margin_pct"]))
    extras = []
    for base in args.also_vs:
        if base in tags and base != args.base:
            extras.append(compare(stats, args.cand, base))

    if args.identity:
        cvs = [st["cv_pct"] for (tag, *_), st in stats.items() if tag in (args.cand, args.base)]
        decision = {
            "keep": False,
            "verdict": "NOISE_STUDY",
            "ok_correct": not fails,
            "ok_fast": False,
            "ok_anchor": True,
            "reasons": [
                f"identity |geo_delta|={abs(primary['geo_delta_pct']):.3f}%",
                f"mean_cv_pct={statistics.mean(cvs):.3f}" if cvs else "no cv",
            ],
            "error_violations": 0,
            "fail_lines": len(fails),
            "abs_geo_delta_pct": abs(primary["geo_delta_pct"]),
            "mean_cv_pct": statistics.mean(cvs) if cvs else float("nan"),
            "bootstrap_identity": boot,
            "min_geo_margin_pct": margin,
        }
    else:
        decision = keep_decision(
            primary,
            rows,
            fails,
            cand=args.cand,
            error_max=args.error_max,
            min_geo_margin_pct=margin,
            require_anchor_win=args.require_anchor_win,
        )
        decision["min_geo_margin_pct"] = margin

    write_outputs(args.out, stats, primary, decision, extras, meta, tags)

    print((args.out / "keep_gate.txt").read_text(), end="")
    print(f"wrote {args.out}/compare.csv keep_gate.json keep_gate.txt shape_stats.csv")
    return 0 if (decision["verdict"] in {"KEEP", "NOISE_STUDY"} or not decision["keep"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
