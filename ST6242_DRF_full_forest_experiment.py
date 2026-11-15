"""Paired bootstrap forests with exact-kernel conditional distribution scoring."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor

from drf_core import gaussian_kernel, forest_weights, mean_mcse, write_csv, write_metadata

SCENARIOS = ["variance", "tail", "bimodality", "dependence"]
METHODS = ["Mean-CART", "RFF-MMD"]


def generate(rng, n, scenario):
    x = rng.uniform(size=(n, 3))
    group = x[:, 0] > 0.5
    if scenario == "dependence":
        z = rng.normal(size=(n, 2))
        rho = np.where(group, -0.8, 0.8)
        y = np.column_stack((z[:, 0], rho * z[:, 0] + 0.6 * z[:, 1]))
    else:
        y = rng.normal(size=n)
        if scenario == "variance":
            y *= np.where(group, 2.0, 1.0)
        elif scenario == "tail":
            y[group] = rng.standard_t(3, size=group.sum()) / np.sqrt(3)
        elif scenario == "bimodality":
            a = 0.95
            y[group] = rng.choice([-a, a], size=group.sum()) + np.sqrt(1 - a*a) * rng.normal(size=group.sum())
        else:
            raise ValueError(scenario)
        y = y[:, None]
    return x, y


def normal_embedding(y, mean, covariance, h):
    """E[k_h(y,Z)] for a multivariate normal Z, evaluated analytically."""
    d = y.shape[1]
    matrix = np.eye(d) + covariance / h**2
    centered = y - mean
    quadratic = np.einsum("ij,jk,ik->i", centered, np.linalg.inv(matrix), centered)
    return np.exp(-quadratic / (2*h*h)) / np.sqrt(np.linalg.det(matrix))


def normal_self(covariance, h):
    return float(np.linalg.det(np.eye(len(covariance)) + 2 * covariance / h**2)**(-0.5))


def student_quantiles(q):
    return (stats.t.ppf((np.arange(q) + 0.5) / q, df=3) / np.sqrt(3))[:, None]


def student_self(quantiles, h):
    total = 0.0
    for start in range(0, len(quantiles), 128):
        total += gaussian_kernel(quantiles[start:start+128], quantiles, h).sum()
    return float(total / len(quantiles)**2)


def law_embeddings(y, scenario, h, quadrature=4096, student_constant=None):
    """Return E[k(y,Z_g)] and E[k(Z_g,Z'_g)] for regimes g=0,1."""
    normal_cross = normal_embedding(y, np.zeros(y.shape[1]), np.eye(y.shape[1]), h)
    normal_pp = normal_self(np.eye(y.shape[1]), h)
    if scenario == "variance":
        return np.column_stack((normal_cross, normal_embedding(y, np.zeros(1), np.array([[4.0]]), h))), np.array([normal_pp, normal_self(np.array([[4.0]]), h)])
    if scenario == "tail":
        quantiles = student_quantiles(quadrature)
        cross = gaussian_kernel(y, quantiles, h).mean(axis=1)
        pp = student_self(quantiles, h) if student_constant is None else student_constant
        return np.column_stack((normal_cross, cross)), np.array([normal_pp, pp])
    if scenario == "bimodality":
        a, variance = 0.95, 1 - 0.95**2
        covariance = np.array([[variance]])
        cross = (normal_embedding(y, np.array([-a]), covariance, h) +
                 normal_embedding(y, np.array([a]), covariance, h)) / 2
        same = normal_self(covariance, h)
        different = same * np.exp(-(2*a)**2 / (2*(h*h + 2*variance)))
        return np.column_stack((normal_cross, cross)), np.array([normal_pp, (same + different)/2])
    covariance0 = np.array([[1.0, 0.8], [0.8, 1.0]])
    covariance1 = np.array([[1.0, -0.8], [-0.8, 1.0]])
    cross = np.column_stack((normal_embedding(y, np.zeros(2), covariance0, h),
                             normal_embedding(y, np.zeros(2), covariance1, h)))
    return cross, np.array([normal_self(covariance0, h), normal_self(covariance1, h)])


def conditional_risk(weights, kernel, groups, cross, pp):
    empirical = np.einsum("ij,ij->i", weights @ kernel, weights)
    return empirical + pp[groups] - 2*np.sum(weights * cross[:, groups].T, axis=1)


def functional_errors(weights, y, groups, scenario):
    means = weights @ y
    if scenario == "variance":
        estimates = weights @ (y[:, 0]**2) - means[:, 0]**2
        truth = np.where(groups, 4.0, 1.0)
    elif scenario == "dependence":
        estimates = weights @ (y[:, 0]*y[:, 1]) - means[:, 0]*means[:, 1]
        truth = np.where(groups, -0.8, 0.8)
    else:
        threshold = 1.0 if scenario == "tail" else 0.5
        estimates = weights @ (np.abs(y[:, 0]) <= threshold)
        p0 = stats.norm.cdf(threshold) - stats.norm.cdf(-threshold)
        if scenario == "tail":
            p1 = stats.t.cdf(np.sqrt(3)*threshold, df=3) - stats.t.cdf(-np.sqrt(3)*threshold, df=3)
        else:
            a, sd = 0.95, np.sqrt(1 - 0.95**2)
            p1 = sum(0.5 * (stats.norm.cdf((threshold-m)/sd) - stats.norm.cdf((-threshold-m)/sd)) for m in [-a, a])
        truth = np.where(groups, p1, p0)
    return np.abs(estimates - truth)


def summarize(records):
    summaries, paired = [], []
    for scenario in SCENARIOS:
        for method in METHODS:
            subset = [r for r in records if r["scenario"] == scenario and r["method"] == method]
            row = {"scenario": scenario, "method": method, "reps": len(subset)}
            for metric in ["mmd", "functional", "heldout"]:
                row[metric+"_mean"], row[metric+"_mcse"] = mean_mcse([r[metric] for r in subset])
            summaries.append(row)
        reference = {r["rep"]: r for r in records if r["scenario"] == scenario and r["method"] == METHODS[0]}
        other = [r for r in records if r["scenario"] == scenario and r["method"] == METHODS[1]]
        for metric in ["mmd", "functional", "heldout"]:
            difference = [r[metric] - reference[r["rep"]][metric] for r in other]
            mean, mcse = mean_mcse(difference)
            paired.append({"scenario": scenario, "metric": metric, "comparison": "RFF-MMD minus Mean-CART",
                           "mean_difference": mean, "paired_mcse": mcse, "reps": len(other)})
    return summaries, paired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag, default in [("reps", 25), ("train", 650), ("test", 360), ("trees", 180), ("features", 48), ("min-leaf", 12), ("seed", 7606242), ("jobs", 2), ("quadrature", 4096)]:
        parser.add_argument("--"+flag, type=int, default=default)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    if args.reps < 2 or min(args.train, args.test, args.trees, args.features, args.min_leaf, args.quadrature) < 1:
        parser.error("Use at least two replications and positive size parameters")
    args.output.mkdir(parents=True, exist_ok=True)
    student_constant = student_self(student_quantiles(args.quadrature), 1.0)
    records, diagnostics, tail_first = [], [], None
    for scenario_id, scenario in enumerate(SCENARIOS):
        h = 1.4 if scenario == "dependence" else 1.0
        for rep in range(args.reps):
            streams = [np.random.default_rng(np.random.SeedSequence([args.seed, scenario_id, rep, j])) for j in range(4)]
            train_x, y = generate(streams[0], args.train, scenario)
            test_x, test_y = generate(streams[1], args.test, scenario)
            groups = (test_x[:, 0] > 0.5).astype(int)
            frequencies = streams[2].normal(size=(args.features, y.shape[1])) / h
            phase = y @ frequencies.T
            features = np.column_stack((np.cos(phase), np.sin(phase))) / np.sqrt(args.features)
            tree_seed = int(streams[3].integers(0, 2**31 - 1))
            kernel = gaussian_kernel(y, y, h)
            cross, pp = law_embeddings(y, scenario, h, args.quadrature, student_constant)
            heldout_kernel = gaussian_kernel(y, test_y, h)
            paired_weights = []
            for method, response in zip(METHODS, [y, features]):
                model = RandomForestRegressor(n_estimators=args.trees, max_features=2,
                        min_samples_leaf=args.min_leaf, bootstrap=True,
                        random_state=tree_seed, n_jobs=args.jobs)
                model.fit(train_x, response[:, 0] if response.shape[1] == 1 else response)
                weights = forest_weights(model, train_x, test_x)
                # An independent check of multiplicities: weighted training responses
                # must reproduce the library's ordinary forest predictions.
                expected = model.predict(test_x).reshape(args.test, -1)
                mismatch = float(np.max(np.abs(weights @ response - expected)))
                if mismatch > 1e-10:
                    raise AssertionError(f"Leaf weights disagree with forest prediction: {mismatch}")
                mmd = conditional_risk(weights, kernel, groups, cross, pp)
                if mmd.min() < -1e-5:
                    raise AssertionError("Unexpectedly negative conditional MMD risk")
                functional = functional_errors(weights, y, groups, scenario)
                empirical = np.einsum("ij,ij->i", weights @ kernel, weights)
                heldout = empirical - 2*np.sum(weights * heldout_kernel.T, axis=1) + 1
                records.append({"scenario": scenario, "rep": rep, "method": method,
                                "tree_seed": tree_seed, "mmd": float(mmd.mean()),
                                "functional": float(functional.mean()), "heldout": float(heldout.mean()),
                                "max_prediction_mismatch": mismatch})
                paired_weights.append(weights)
            if scenario == "tail" and rep == 0:
                tail_first = (y, kernel, groups, paired_weights)
            # Persist each completed paired replicate so interrupted runs retain evidence.
            write_csv(args.output / "drf_full_forest_replicates.csv", records)
            print(f"{scenario}: paired repetition {rep+1}/{args.reps}", flush=True)
    summaries, paired = summarize(records)
    write_csv(args.output / "drf_full_forest_summary.csv", summaries)
    write_csv(args.output / "drf_full_forest_paired.csv", paired)

    # Check quadrature with the first tail replicate's fixed fitted estimators.
    y, kernel, groups, weights_list = tail_first
    grids = sorted(set([1024, 2048, 4096, 8192, args.quadrature]))
    risks = {}
    for q in grids:
        cross, pp = law_embeddings(y, "tail", 1.0, q)
        risks[q] = [conditional_risk(w, kernel, groups, cross, pp) for w in weights_list]
    reference_q = max(grids)
    for q in grids:
        for j, method in enumerate(METHODS):
            diagnostics.append({"scenario": "tail", "rep": 0, "method": method,
                                "quadrature": q, "reference_quadrature": reference_q,
                                "mmd": float(risks[q][j].mean()),
                                "absolute_mean_change": float(abs(risks[q][j].mean() - risks[reference_q][j].mean())),
                                "max_query_change": float(np.max(np.abs(risks[q][j] - risks[reference_q][j])))})
    write_csv(args.output / "drf_quadrature_check.csv", diagnostics)

    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.3))
    x = np.arange(4)
    for j, method in enumerate(METHODS):
        rows = [next(r for r in summaries if r["scenario"] == s and r["method"] == method) for s in SCENARIOS]
        axes[0, 0].bar(x + (j-.5)*.32, [r["mmd_mean"] for r in rows], width=.32,
                       yerr=[1.96*r["mmd_mcse"] for r in rows], capsize=2, label=method)
    axes[0, 0].set_xticks(x, ["Variance", "Tail", "Modality", "Dependence"], rotation=25)
    axes[0, 0].set_ylabel("Conditional squared MMD")
    axes[0, 0].legend(fontsize=7)
    axes[0, 1].axis("off")
    axes[0, 1].text(0, .8, "Primary distributional metric above.\nFunctional errors have separate scales.\nError bars: 1.96 MCSE.\n25 paired datasets per setting.".replace("25 paired", f"{args.reps} paired"), va="top", fontsize=10)
    axes[0, 2].axis("off")
    functional_axes = [axes[1, 0], axes[1, 1], axes[1, 2], axes[0, 2]]
    for ax, scenario, title in zip(functional_axes, SCENARIOS, ["Variance MAE", "Central-probability MAE", "Modality-probability MAE", "Covariance MAE"]):
        ax.axis("on")
        rows = [next(r for r in summaries if r["scenario"] == scenario and r["method"] == m) for m in METHODS]
        ax.bar([0, 1], [r["functional_mean"] for r in rows], color=["tab:blue", "tab:orange"],
               yerr=[1.96*r["functional_mcse"] for r in rows], capsize=3)
        ax.set_xticks([0, 1], METHODS, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(args.output / "drf_full_forest_scores.pdf")
    plt.close(fig)
    write_metadata(args.output / "full_forest_metadata.json", args, [__file__, Path(__file__).with_name("drf_core.py")])
    print("Full-forest experiments, paired contrasts, and quadrature check saved.", flush=True)


if __name__ == "__main__":
    main()
