"""Run the returned reviewer comparator for exactly one new cohort against OLD_P12."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent


if __name__=="__main__":
    args=argparse.ArgumentParser()
    args.add_argument("condition",choices=("NEW_P12","NEW_D24_B12"))
    name=args.parse_args().condition
    slug={"NEW_P12":"new-p12","NEW_D24_B12":"new-d24-b12"}[name]
    old=ROOT.parent/"preupgrade-baseline-20261123/trial/runs"
    new=ROOT/"trial/runs"
    tool=ROOT.parent/"postload-review-20261124/reviews/measurement_audit/compare_saved_runs.py"
    cmd=[sys.executable,"-I",str(tool),"--reference-quality",str(old/"old-p12-quality-01"),
         "--candidate-quality",str(new/(slug+"-quality-01")),"--cases",str(ROOT/"trial/CASES.json"),
         "--out",str(ROOT/("COMPARE_"+name+".json"))]
    for i in (1,2):
        cmd.extend(["--timing-run",str(new/(slug+f"-timing-0{i}")),"--reference-timing-run",str(old/f"old-p12-timing-0{i}")])
    raise SystemExit(subprocess.run(cmd).returncode)
