"""Run the missing fixed-cache oracle intervals; reuse the completed interval-3 run."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SWEEP = Path(__file__).resolve().parent
RUNNER = ROOT / "research/results/oracle_20260911/run_oracle.py"

for interval in (2, 4, 5):
    output = SWEEP / f"interval_{interval}"
    if (output / "complete.json").exists():
        print(f"Skipping completed {output}", flush=True)
        continue
    if output.exists():
        raise RuntimeError(f"Partial output exists: {output}; inspect it before retrying")
    subprocess.run([sys.executable, "-B", str(RUNNER), "--interval", str(interval),
                    "--output", str(output)], cwd=ROOT, check=True)
