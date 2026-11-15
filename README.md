# ST6242: Distributional random forest experiments

Companion experiments for Le Thi Hong Ha's final project on distributional random forests. The repository includes executable simulations, per-replication records, Monte Carlo summaries, paired method contrasts, a Student-law quadrature check, figures, and the revised LaTeX report.

## Provenance

The original draft's scripts were unavailable. These scripts implement its stated designs anew. The root-split and bandwidth runs reproduce all numbers printed in the original draft at the reported precision. The full-forest study was rerun with the explicit seed scheme below; the revised report uses this run's results throughout. The earlier full-forest numbers must not be attributed to these scripts.

The full-forest experiment uses fixed random Fourier features and ordinary non-honest bootstrap trees. It is an approximation to the MMD splitting principle. It does not implement the official honest DRF algorithm or half-sampling inference and does not empirically validate their asymptotic guarantees.

## Reproduce

Use Python 3.10 with the versions in `requirements.txt`. The supplied versions match the recorded execution environment. Numerical details can vary with other library versions or platforms.

```sh
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python ST6242_DRF_extended_experiment.py
python ST6242_DRF_full_forest_experiment.py
python build_report_inputs.py
cd report
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The experiments do not download datasets or require an account. All data are synthetic. Outputs go to `results/`. Both scripts support `--help`; use a different `--output` directory for exploratory runs so the supplied results stay intact. `build_report_inputs.py` generates table rows and numeric macros from CSVs and copies the matching figures into `report/`.

For a quick execution check, without replacing the reported results:

```sh
python ST6242_DRF_extended_experiment.py --reps 3 --bandwidth-reps 3 --output /tmp/drf-root-smoke
python ST6242_DRF_full_forest_experiment.py --reps 2 --train 80 --test 20 --trees 5 --features 8 --min-leaf 5 --output /tmp/drf-forest-smoke
```

If using the original Mac's Intel Python installation on Apple Silicon, prefix commands with `arch -x86_64`. This is specific to that installation, not a requirement of the code. Setting `MPLCONFIGDIR` to a writable directory avoids Matplotlib cache warnings. To limit numerical-library threads, set `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, and `VECLIB_MAXIMUM_THREADS=1` when running the scripts. The forest defaults to two workers and accepts `--jobs`.

## Designs

The root-split study uses 200 independent datasets at each of `n = 200, 400, 800`, with one sequential NumPy `default_rng(6242)` stream. All methods use each dataset, and each child must contain at least 15% of observations. Responses have mean zero and variance 1 or 4 across the threshold 0.5. Comparisons are CART on Y, CART on Y squared, and exact Gaussian-kernel MMD with bandwidth 1. A separate stream, `default_rng(6243)`, generates 120 datasets of size 400 for bandwidths 0.25, 0.5, 1, 2, and 4.

The full-forest study uses 25 repetitions per setting; 650 training observations; 360 independent test pairs; 180 bootstrap trees; 48 Fourier frequencies, giving 96 sine/cosine outputs; `max_features=2`; and `min_samples_leaf=12`. Covariates are independent uniforms in three dimensions; the first coordinate determines the regime. Settings vary variance, tail shape at equal variance, modality at equal first two moments, and bivariate dependence at equal marginal laws. Bandwidth is 1 for scalar responses and 1.4 for the dependence setting.

For full-forest scenario index `s` in `[variance, tail, bimodality, dependence]` and zero-based repetition `r`, independent streams use `SeedSequence([7606242, s, r, j])`, with `j=0` for training data, `j=1` for test data, `j=2` for Fourier features, and `j=3` for the common forest seed. Each paired fit uses the same covariates, responses, bootstrap seed, and tree parameters. The bootstrap counts are reconstructed from each tree's stored random seed. Weighted leaf predictions are checked against scikit-learn's predictions in every fit.

## Scores and uncertainty

The primary metric is mean conditional squared MMD between the weighted empirical forest law and the known true law, using exact Gaussian kernel values. Normal and Gaussian-mixture expectations are analytic; the Student law uses 4096 deterministic midpoint quantiles. The first fitted tail replicate is also scored with 1024, 2048, 4096, and 8192 points, keeping fitted estimators fixed. This is a numerical convergence diagnostic, not a uniform integration-error bound.

Functional errors target variance, a central probability, a modality-sensitive probability, and covariance. Their scales differ. The held-out kernel score in the CSVs is the squared MMD to each observed test response's point mass. Its expectation equals the true-law MMD risk plus a law-dependent constant, so it is a complementary prediction score.

MCSE is the sample standard deviation across independent datasets divided by the square root of the number of datasets. Paired files compute this quantity from within-dataset error differences. In the full-forest contrast, a negative difference favors RFF-MMD. Query points and trees are not independent Monte Carlo replicates. Error bars on individual-method plots do not replace the paired contrasts.

## Files

- `ST6242_DRF_extended_experiment.py`: root-split and bandwidth experiments and three figures.
- `ST6242_DRF_full_forest_experiment.py`: paired forest experiments, quadrature check, and comparison figure.
- `drf_core.py`: exact MMD, forest weights, summaries, metadata.
- `tests/test_experiments.py`: numerical checks against direct calculations, integration, and library predictions.
- `build_report_inputs.py`: CSV-to-LaTeX tables and numeric macros.
- `results/*_metadata.json`: parameters, package versions, and SHA-256 hashes of experiment source files.
- `results/*.csv`: per-replication records, method summaries, paired contrasts, and quadrature checks.
- `results/*.pdf`: generated figures.
- `report/`: revised standalone LaTeX report and generated inputs.

The project code and records are distinct from the authors' implementations: [DRF package and applications](https://github.com/lorismichel/drf) and [2023 inference-paper experiments](https://github.com/JeffNaef/drfinference). These links are also supplied in the report's reproducibility appendix.

## Publish to your GitHub account

This directory is ready to upload as a repository. It contains no GitHub credentials and no assumed owner or remote URL. Include `results/` and `report/` so the checked numerical evidence accompanies the code. A hosted repository URL is available only after repository creation and upload to your account.
