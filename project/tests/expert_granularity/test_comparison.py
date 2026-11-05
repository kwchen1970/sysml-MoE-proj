"""CPU-only comparison test; synthetic traces stay in a temporary directory."""
import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
ANALYSIS = ROOT / "project/analysis/expert_granularity"

class ComparisonTests(unittest.TestCase):
    def test_rank_aggregation_and_all_figure_formats(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "csrc/cmake-build-release") as d:
            d=Path(d)
            rows=[]
            for ngpu in (1,4):
                for rep in range(2):
                    for rank in range(ngpu):
                        start=10**16 + rank*10**15
                        base=dict(run_id="fixture",invocation=f"{ngpu}-{rep}",ngpu=ngpu,E=16,rank=rank,
                                  block_id=0,sm_id=rank,role="processor",window_start_ns=start,
                                  window_end_ns=start+100,dropped_events=0,task_type=0,tile_id=0)
                        for state,begin,end in [("other",0,25),("wait",25,50),("task",50,100)]:
                            rows.append(dict(base,state=state,start_ns=start+begin,end_ns=start+end))
            events=d/"events.csv"
            with events.open("w",newline="") as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            output=d/"figures"
            subprocess.run([sys.executable,str(ANALYSIS/"compare_traces.py"),str(events),"--output",str(output)],check=True)
            with (output/"block_time_breakdown.csv").open() as f:
                summaries=list(csv.DictReader(f))
            self.assertEqual(len(summaries),2)
            for summary in summaries:
                self.assertEqual(int(summary["invocations"]),2)
                self.assertEqual(float(summary["wait_mean_pct"]),25)
            with (output/"task_metrics.csv").open() as f:
                metrics=list(csv.DictReader(f))
            for metric in metrics:
                self.assertEqual(float(metric["count_per_gpu"]),1)
                self.assertEqual(float(metric["mean_us"]),.05)
            for stem in ("block_time_breakdown","task_duration_comparison","task_count_comparison","per_rank_waiting"):
                ET.parse(output/(stem+".svg"))
                self.assertTrue((output/(stem+".png")).is_file())
                self.assertTrue((output/(stem+".pdf")).is_file())

if __name__=="__main__": unittest.main()
