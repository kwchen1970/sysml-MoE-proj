"""CPU regression checks. Synthetic inputs remain in temporary directories."""
import csv
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
ANALYSIS = PROJECT / "analysis/expert_granularity"
sys.path.insert(0, str(ANALYSIS))
sys.path.insert(0, str(PROJECT / "scripts/expert_granularity"))
from intervals import load
from summarize import summarize
from run import FIELDS, parse_output

def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

class PipelineTests(unittest.TestCase):
    def test_rank_validation(self):
        row = dict.fromkeys(FIELDS, "0")
        row.update(S="4096", H="2048", I="1408", E="16", k="6", time_ms="1")
        text = ",".join(row[x] for x in FIELDS)
        self.assertEqual(len(parse_output(text, 1, (4096,2048,1408,16,6))), 1)
        for bad in ("", text+"\n"+text):
            with self.assertRaises(ValueError):
                parse_output(bad, 1, (4096,2048,1408,16,6))

    def test_max_rank_before_repetition_statistics(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"measurements.csv"
            rows=[]
            for inv,times in enumerate(([1,2,3,4],[2,3,4,6])):
                for rank,t in enumerate(times):
                    rows.append(dict(run_id="test",invocation=inv,ngpu=4,S=4096,H=2048,I=1408,E=16,k=6,rank=rank,time_ms=t,error_pct=0))
            write_csv(p,rows)
            result,_=summarize(p)
            self.assertEqual(result[0]["mean_ms"],5)
            self.assertEqual(result[0]["n"],2)
            write_csv(p,rows[:-1])
            with self.assertRaises(ValueError): summarize(p)

    def test_intervals_analysis_and_plots(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            # Deliberately beyond exact float integer range: preserve 64-bit ns.
            base=10**16
            common=dict(run_id="test",invocation="0",ngpu=1,E=16,rank=0,block_id=0,sm_id=1,role="processor",
                        window_start_ns=base,window_end_ns=base+100,dropped_events=0,task_type=0,tile_id=0)
            rows=[dict(common,state="wait",start_ns=base,end_ns=base+25),
                  dict(common,state="task",start_ns=base+25,end_ns=base+100)]
            p=d/"events.csv";write_csv(p,rows)
            self.assertEqual(list(load(p).values())[0][0]["end_ns"],base+25)
            subprocess.run([sys.executable,str(ANALYSIS/"waiting.py"),str(p),"--output",str(d/"out")],check=True)
            subprocess.run([sys.executable,str(ANALYSIS/"timeline.py"),str(p),"--run-id","test","--invocation","0",
                            "--duration-us","1","--output",str(d/"timeline.svg")],check=True)
            with (d/"out/waiting.csv").open() as f:
                self.assertEqual(float(next(csv.DictReader(f))["waiting_fraction"]),.25)
            with (d/"out/work_bound.csv").open() as f:
                result=next(csv.DictReader(f))
                self.assertAlmostEqual(float(result["captured_actor_span_ms"]),.0001)
                self.assertAlmostEqual(float(result["task_work_bound_ms"]),.000075)
            for svg in (d/"timeline.svg",d/"out/waiting_vs_experts.svg",d/"out/span_vs_work_bound.svg"):
                ET.parse(svg)
                self.assertTrue(svg.with_suffix(".png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
                self.assertTrue(svg.with_suffix(".pdf").read_bytes().startswith(b"%PDF"))
            rows[1]["start_ns"]+=1;write_csv(p,rows)
            with self.assertRaises(ValueError): load(p)
            rows[1]["start_ns"]-=1;rows[0]["dropped_events"]=1;write_csv(p,rows)
            with self.assertRaises(ValueError): load(p)

if __name__=="__main__":
    unittest.main()
