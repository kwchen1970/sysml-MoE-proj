"""CPU-only launcher tests: srun is a stub, never Slurm."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
LAUNCH = ROOT / "project/scripts/expert_granularity/launch_single_node.sh"

class LaunchTests(unittest.TestCase):
    def invoke(self, count):
        # Fixtures remain inside the repository and are removed by TemporaryDirectory.
        with tempfile.TemporaryDirectory(dir=ROOT / "csrc/cmake-build-release") as d:
            stub=Path(d)/"srun"
            stub.write_text("#!/usr/bin/env python3\nimport json,os,sys\nprint(json.dumps(dict(args=sys.argv[1:],visible=os.environ.get('CUDA_VISIBLE_DEVICES'))))\n")
            stub.chmod(0o700)
            env=os.environ.copy()
            env["PATH"]=d+os.pathsep+str(Path(os.sys.executable).parent)+os.pathsep+env["PATH"]
            env["CUDA_VISIBLE_DEVICES"]="GPU-a,GPU-b,GPU-c,GPU-d"
            result=subprocess.run(["bash",str(LAUNCH),str(count),"testFlashMoE","4096"],
                                  env=env,text=True,capture_output=True,check=True)
            return json.loads(result.stdout)

    def test_four_gpus_share_allocated_visibility(self):
        result=self.invoke(4)
        self.assertIn("--gpus-per-node=4",result["args"])
        self.assertIn("--gpu-bind=none",result["args"])
        self.assertIn("--ntasks-per-node=4",result["args"])
        self.assertFalse(any(x.startswith("--gpus-per-task") for x in result["args"]))
        self.assertEqual(result["visible"],"GPU-a,GPU-b,GPU-c,GPU-d")

    def test_single_gpu_preserves_existing_binding(self):
        result=self.invoke(1)
        self.assertIn("--gpus-per-task=1",result["args"])
        self.assertIn("--gpu-bind=closest",result["args"])
        self.assertEqual(result["args"][-2:],["testFlashMoE","4096"])

    def test_rejects_unsupported_counts(self):
        result=subprocess.run(["bash",str(LAUNCH),"2","testFlashMoE"],capture_output=True)
        self.assertEqual(result.returncode,2)

if __name__=="__main__":
    unittest.main()
