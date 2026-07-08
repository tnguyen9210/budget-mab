# budget-mab

KUBE / Fractional KUBE for budget-limited multi-armed bandits
(Tran-Thanh et al. 2012, arXiv:1204.1909). See `paper.pdf` and
`docs/algorithms.md` for the algorithm/code mapping, `docs/issues.md`
for known deviations from the paper's Section 5 setup.

## Running an experiment

All commands run from the repo root. Config is Hydra-based (same
pattern as llm-reasoning-methods): `conf/<name>.yaml` presets,
selected with `--config-name`, and any field overridden inline with
`key=value`:

```bash
python run_experiments.py --config-name default
python plot_results.py --results results/default.pkl --out_dir results
```

```bash
python run_experiments.py --config-name default n_seeds=10
python run_experiments.py --config-name smoke regimes=[homogeneous]
```

`conf/smoke.yaml` is a tiny fast config for checking the pipeline
runs end-to-end before committing to a full sweep. The schema lives
in `src/exp_config.py` (`ExpConfig`).

Each run writes `results/<name>.pkl` (the data) and
`results/<name>.json` (a provenance sidecar: git commit, the config
used, and resolved parameters) — check the sidecar before comparing
two `.pkl` files, since results generated under different code/
parameters are not comparable even if they share a filename pattern.

`results/` is gitignored; nothing there is committed.
