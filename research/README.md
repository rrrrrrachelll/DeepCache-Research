# Research experiments

Run commands below from the repository root. The result directories keep the experiment scripts and tests alongside their images and measurements; their reports or protocols explain each run.

| Experiment | Directory | Main command |
| --- | --- | --- |
| Sampling and fixed-cache pilot | [pilot_20260908](results/pilot_20260908/) | `python -B research/results/pilot_20260908/compare_sampling.py --output research/results/new_pilot_run` |
| Adaptive V1, threshold 0.65 | [adaptive_v1_20260909](results/adaptive_v1_20260909/) | `python -B research/results/adaptive_v1_20260909/run_adaptive.py --output research/results/new_adaptive_run` |
| Adaptive V1, threshold 0.50 | [adaptive_v1_tau050_20260909](results/adaptive_v1_tau050_20260909/) | `python -B research/results/adaptive_v1_tau050_20260909/run.py --output research/results/new_tau050_run` |
| Oracle cache analysis | [oracle_20260911](results/oracle_20260911/) | `python -B research/results/oracle_20260911/run_oracle.py --output research/results/new_oracle_run` |
| Oracle interval sweep (2, 3, 4, 5) | [oracle_interval_sweep_20260914](results/oracle_interval_sweep_20260914/) | See the completed `REPORT.md` and reproducible `run_sweep.py` |

The two adaptive runs share the controller and sampling utilities. The threshold-0.50 directory has its own entry point that selects that experiment's setting.

Historical `manifest.json` files retain their original source paths and SHA256 values as run records; those paths refer to the pre-organization layout. New runs record the current paths.
