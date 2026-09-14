"""Reproduce the adaptive V1 threshold-0.50 run with shared V1 code."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from research.results.adaptive_v1_20260909.run_adaptive import main


if __name__ == "__main__":
    if not any(arg == "--risk-threshold" or arg.startswith("--risk-threshold=")
               for arg in sys.argv[1:]):
        sys.argv.extend(["--risk-threshold", "0.50"])
    main()
