"""Numerical primitives shared by the ST6242 experiments."""
from pathlib import Path
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import warnings

import numpy as np
from scipy.spatial.distance import cdist


# \linewidth of the report (a4paper, 1in margins) in inches. Figures are drawn
# at the width at which they are included, so fonts print at their nominal size.
TEXT_WIDTH_IN = 452.9679 / 72.27


def report_style():
    """Matplotlib settings matching the report: LaTeX with Latin Modern (lmodern, T1).

    Without a LaTeX installation (for example on a CI runner), or with
    DRF_NO_USETEX=1, figures fall back to Matplotlib's built-in mathtext with
    Computer Modern math. Results are unaffected; only the figure fonts differ.
    """
    import matplotlib.pyplot as plt
    settings = {
        "font.family": "serif",
        "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "legend.handlelength": 1.6, "legend.borderaxespad": 0.4,
        "pdf.fonttype": 42,
    }
    if shutil.which("latex") and os.environ.get("DRF_NO_USETEX") != "1":
        settings.update({
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{amsmath}",
        })
    else:
        warnings.warn("LaTeX not found or disabled; figures use mathtext instead of the report fonts.")
        settings.update({"text.usetex": False, "mathtext.fontset": "cm"})
    plt.rcParams.update(settings)


def gaussian_kernel(y, z, bandwidth=1.0):
    y = np.asarray(y).reshape(len(y), -1)
    z = np.asarray(z).reshape(len(z), -1)
    return np.exp(-cdist(y, z, metric="sqeuclidean") / (2 * bandwidth**2))


def candidate_splits(x, min_fraction=0.15):
    """Return the sort order, permissible left counts, and midpoint thresholds."""
    order = np.argsort(x, kind="stable")
    xs = np.asarray(x)[order]
    minimum = int(np.ceil(min_fraction * len(xs)))
    counts = np.arange(minimum, len(xs) - minimum + 1)
    counts = counts[xs[counts - 1] < xs[counts]]
    if not len(counts):
        raise ValueError("No permissible split between distinct covariate values")
    return order, counts, (xs[counts - 1] + xs[counts]) / 2


def cart_scores(sorted_y, left_counts):
    y = np.asarray(sorted_y).reshape(len(sorted_y), -1)
    n = len(y)
    total = y.sum(axis=0)
    left = y.cumsum(axis=0)[left_counts - 1]
    difference = left / left_counts[:, None] - (total - left) / (n - left_counts)[:, None]
    return left_counts * (n - left_counts) / n**2 * (difference**2).sum(axis=1)


def mmd_split_scores(sorted_y, left_counts, bandwidth=1.0):
    """All balanced biased-MMD scores from cumulative kernel block sums."""
    kernel = gaussian_kernel(sorted_y, sorted_y, bandwidth)
    block = kernel.cumsum(axis=0).cumsum(axis=1)
    n = len(kernel)
    ll = block[left_counts - 1, left_counts - 1]
    all_left = block[left_counts - 1, -1]
    lr = all_left - ll
    rr = block[-1, -1] - ll - 2 * lr
    right_counts = n - left_counts
    return (right_counts / left_counts * ll + left_counts / right_counts * rr - 2 * lr) / n**2


def forest_weights(forest, train_x, query_x):
    """Use in-bag bootstrap multiplicities, including zero weight for OOB points.

    RandomForestRegressor samples n training indices using each tree's stored
    random_state when bootstrap=True and max_samples=None. Reconstructing those
    draws uses NumPy's RandomState, the same generator used by scikit-learn.
    """
    if not forest.bootstrap or forest.max_samples is not None:
        raise ValueError("This experiment requires bootstrap=True, max_samples=None")
    n = len(train_x)
    weights = np.zeros((len(query_x), n))
    for tree in forest.estimators_:
        draws = np.random.RandomState(tree.random_state).randint(0, n, n)
        counts = np.bincount(draws, minlength=n)
        train_leaf = tree.apply(train_x)
        query_leaf = tree.apply(query_x)
        for leaf in np.unique(query_leaf):
            indices = np.flatnonzero((train_leaf == leaf) & (counts > 0))
            leaf_counts = counts[indices]
            if not len(indices):
                raise AssertionError("A query reached an empty bootstrap leaf")
            rows = np.flatnonzero(query_leaf == leaf)
            weights[np.ix_(rows, indices)] += leaf_counts / leaf_counts.sum()
    weights /= len(forest.estimators_)
    if np.any(weights < 0) or not np.allclose(weights.sum(axis=1), 1.0, atol=1e-12):
        raise AssertionError("Invalid distributional forest weights")
    return weights


def mean_mcse(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_metadata(path, arguments, scripts):
    versions = {name: importlib.metadata.version(name)
                for name in ["numpy", "scipy", "scikit-learn", "matplotlib", "joblib", "threadpoolctl"]}
    payload = {"python": platform.python_version(), "platform": platform.platform(),
               "packages": versions, "arguments": vars(arguments),
               "source_sha256": {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                 for p in scripts},
               "provenance": "New implementation of the draft's stated designs; original scripts were unavailable."}
    Path(path).write_text(json.dumps(payload, indent=2, default=str) + "\n")
