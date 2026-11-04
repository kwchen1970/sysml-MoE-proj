"""Validate the device-recorder interchange schema. No synthetic/fallback events."""
import csv, math
from collections import defaultdict
KEYS=("run_id","invocation","ngpu","E","rank","block_id","sm_id","role")
def load(path):
    groups=defaultdict(list)
    required=set(KEYS)|{"start_ns","end_ns","window_start_ns","window_end_ns","state","dropped_events"}
    with open(path) as f:
        reader=csv.DictReader(f)
        if not required <= set(reader.fieldnames or []):
            raise ValueError("Missing trace columns: "+", ".join(sorted(required-set(reader.fieldnames or []))))
        for r in reader:
            for k in ("ngpu","E","rank","block_id","sm_id","dropped_events"): r[k]=int(r[k])
            for k in ("start_ns","end_ns","window_start_ns","window_end_ns"):
                r[k]=int(r[k])
                if not math.isfinite(r[k]): raise ValueError("Non-finite trace timestamp")
            if r["dropped_events"]: raise ValueError("Trace overflow: waiting fractions would be unreliable")
            if not r["window_start_ns"] <= r["start_ns"] < r["end_ns"] <= r["window_end_ns"]: raise ValueError("Invalid event/window bounds")
            if r["role"]=="processor" and r["state"] not in {"wait","task","other"}: raise ValueError("Processor states must be wait/task/other")
            groups[tuple(r[k] for k in KEYS)].append(r)
    if not groups: raise ValueError("Empty trace")
    for rows in groups.values():
        rows.sort(key=lambda r:r["start_ns"])
        window=(rows[0]["window_start_ns"],rows[0]["window_end_ns"])
        for i,r in enumerate(rows):
            if (r["window_start_ns"],r["window_end_ns"])!=window: raise ValueError("Inconsistent lane windows")
            if i and r["start_ns"]<rows[i-1]["end_ns"]: raise ValueError("Overlapping lane intervals")
        if rows[0]["role"]=="processor":
            if rows[0]["start_ns"]!=window[0] or rows[-1]["end_ns"]!=window[1] or any(a["end_ns"]!=b["start_ns"] for a,b in zip(rows,rows[1:])):
                raise ValueError("Processor intervals must partition the entire declared window; include other/startup/drain explicitly")
    return groups
