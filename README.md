# Distributional Random Forests: Experiments

Code and figures for my ST6242 final project on distributional random forests (DRF). The experiments compare mean-based and MMD-based splitting when the conditional distribution changes but the conditional mean does not.

The forests here use fixed random Fourier features and ordinary (non-honest) bootstrap trees, so they approximate MMD splitting rather than implement the official DRF algorithm. For the authors' code, see the [DRF package](https://github.com/lorismichel/drf) and the [inference-paper experiments](https://github.com/JeffNaef/drfinference).

## Running

Python 3.10 with the versions in `requirements.txt`. All data are simulated; nothing is downloaded.

```sh
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python root_split_experiment.py         # root splits and bandwidths
python full_forest_experiment.py        # full forests, four designs
python smooth_transition_experiment.py  # smooth variance transitions
```

Each script writes per-repetition CSVs, summaries, paired contrasts, metadata, and figures to `results/`; only the figures are tracked in the repository. Each script accepts `--help`; pass a different `--output` for exploratory runs.

Figures are typeset with LaTeX to match the report fonts. Without a TeX installation (or with `DRF_NO_USETEX=1`), Matplotlib's built-in math rendering is used instead; the numbers are unaffected. This is how the GitHub Actions smoke tests run:

```sh
python root_split_experiment.py --reps 3 --bandwidth-reps 3 --output /tmp/drf-root-smoke
python full_forest_experiment.py --reps 2 --train 80 --test 20 --trees 5 --features 8 --min-leaf 5 --output /tmp/drf-forest-smoke
python smooth_transition_experiment.py --reps 2 --train 80 --test 20 --trees 5 --features 8 --min-leaf 5 --output /tmp/drf-smooth-smoke
```

## Experiments

**Root split.** `Y | X=x` is `N(0,1)` for `x <= 0.5` and `N(0,4)` otherwise. For 200 datasets at each `n = 200, 400, 800`, I compare the root split chosen by CART on `Y`, CART on `Y^2`, and exact Gaussian-kernel MMD (bandwidth 1). A second run uses 120 datasets of size 400, bandwidths 0.25 to 32, and both CART rules on the same datasets, recording how often the MMD split coincides with each CART split. Seeds: `default_rng(6242)` and `default_rng(6243)`.

**Full forests.** Four designs change the variance, the tail shape, the modality, or the bivariate dependence at `x1 = 0.5`, with three uniform covariates. Each repetition fits a mean-CART forest and an RFF-MMD forest (48 frequencies) on the same data and bootstrap draws: 650 training points, 360 test points, 180 trees, `max_features=2`, `min_samples_leaf=12`, 25 repetitions per design. Seeds: `SeedSequence([7606242, s, r, j])` for design `s`, repetition `r`, and stream `j` (0 training, 1 test, 2 Fourier features, 3 forest).

**Smooth transitions.** The variance step is replaced by `Var(Y | X=x) = 1 + 3 / (1 + exp(-(x1 - 0.5)/eps))` for `eps` in `step, 0.025, 0.05, 0.1, 0.2`, with the same forest settings. Seeds: `SeedSequence([7606243, e, r, j])`.

## Metrics

- **Conditional MMD².** Squared MMD between each forest's weighted empirical law and the true conditional law, computed exactly (the Student law uses 4096-point quadrature, checked against 8192).
- **Functional error.** Absolute error for one target per design: variance, a central probability, a modality-sensitive probability, or covariance.
- **Decomposition.** The MMD² is split into localization bias (`m^2 D_h` for step designs, where `m` is the weight on the wrong regime) and a remainder. The noise proxy and effective sample size are recorded alongside.
- **Uncertainty.** Monte Carlo standard errors across datasets, and paired differences (RFF-MMD minus mean-CART; negative favors RFF-MMD).

## Files

- `drf_core.py`: kernels, MMD split scores, forest weights, summaries, figure style.
- `root_split_experiment.py`, `full_forest_experiment.py`, `smooth_transition_experiment.py`: the three experiments.
- `tests/`: checks against direct calculation, numerical integration, and scikit-learn predictions.
- `results/`: the figures used in the report.
