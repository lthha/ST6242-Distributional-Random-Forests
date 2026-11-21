"""Paired forests under a smooth variance transition of width epsilon.

Y | X = x ~ N(0, v_eps(x1)), v_eps(t) = 1 + 3 / (1 + exp(-(t - 1/2) / eps)),
with eps = 0 denoting the step change of the full-forest variance scenario.
For every eps > 0 the conditional kernel mean embedding is Lipschitz in x, so
the design satisfies the smoothness assumption of the DRF consistency and
inference theorems, while eps -> 0 recovers the discontinuous design.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from drf_core import gaussian_kernel, forest_weights, mean_mcse, write_csv, write_metadata, report_style, TEXT_WIDTH_IN

METHODS = ["Mean-CART", "RFF-MMD"]
EPSILONS = [0.0, 0.025, 0.05, 0.1, 0.2]
METRICS = ["mmd", "variance_mae", "bias", "remainder", "n_eff"]


def variance_function(t, eps):
    if eps == 0:
        return np.where(t > 0.5, 4.0, 1.0)
    return 1.0 + 3.0 / (1.0 + np.exp(-(t - 0.5) / eps))


def generate(rng, n, eps):
    x = rng.uniform(size=(n, 3))
    v = variance_function(x[:, 0], eps)
    return x, rng.normal(size=n) * np.sqrt(v), v


def normal_inner(va, vb, h):
    """<mu_{N(0,va)}, mu_{N(0,vb)}> for the Gaussian kernel with bandwidth h."""
    return (1.0 + (va + vb) / h**2) ** -0.5


def conditional_risk(weights, kernel, y, v_train, v_test, h):
    """Exact MMD^2 between the weighted empirical law and N(0, v(x)) per query,
    with its split into localisation bias ||sum_i w_i mu(X_i) - mu(x)||^2 and
    the remainder."""
    empirical = np.einsum("ij,ij->i", weights @ kernel, weights)
    cross = (1.0 + v_test[:, None] / h**2) ** -0.5 * np.exp(-y[None, :]**2 / (2*(h**2 + v_test[:, None])))
    self_term = normal_inner(v_test, v_test, h)
    risk = empirical + self_term - 2*np.sum(weights * cross, axis=1)
    train_gram = normal_inner(v_train[:, None], v_train[None, :], h)
    test_train = normal_inner(v_test[:, None], v_train[None, :], h)
    bias = (np.einsum("ij,ij->i", weights @ train_gram, weights) + self_term
            - 2*np.sum(weights * test_train, axis=1))
    return risk, bias


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag, default in [("reps", 25), ("train", 650), ("test", 360), ("trees", 180), ("features", 48),
                          ("min-leaf", 12), ("seed", 7606243), ("jobs", 2)]:
        parser.add_argument("--"+flag, type=int, default=default)
    parser.add_argument("--bandwidth", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    if args.reps < 2 or min(args.train, args.test, args.trees, args.features, args.min_leaf) < 1:
        parser.error("Use at least two replications and positive size parameters")
    args.output.mkdir(parents=True, exist_ok=True)
    h = args.bandwidth
    records = []
    for eps_id, eps in enumerate(EPSILONS):
        for rep in range(args.reps):
            streams = [np.random.default_rng(np.random.SeedSequence([args.seed, eps_id, rep, j])) for j in range(4)]
            train_x, y, v_train = generate(streams[0], args.train, eps)
            test_x, _, v_test = generate(streams[1], args.test, eps)
            frequencies = streams[2].normal(size=args.features) / h
            phase = np.outer(y, frequencies)
            features = np.column_stack((np.cos(phase), np.sin(phase))) / np.sqrt(args.features)
            tree_seed = int(streams[3].integers(0, 2**31 - 1))
            kernel = gaussian_kernel(y, y, h)
            for method, response in zip(METHODS, [y, features]):
                model = RandomForestRegressor(n_estimators=args.trees, max_features=2,
                                              min_samples_leaf=args.min_leaf, bootstrap=True,
                                              random_state=tree_seed, n_jobs=args.jobs)
                model.fit(train_x, response)
                weights = forest_weights(model, train_x, test_x)
                expected = model.predict(test_x).reshape(args.test, -1)
                mismatch = float(np.max(np.abs(weights @ response.reshape(args.train, -1) - expected)))
                if mismatch > 1e-10:
                    raise AssertionError(f"Leaf weights disagree with forest prediction: {mismatch}")
                risk, bias = conditional_risk(weights, kernel, y, v_train, v_test, h)
                mean = weights @ y
                variance_error = np.abs(weights @ y**2 - mean**2 - v_test)
                records.append({"epsilon": eps, "rep": rep, "method": method, "tree_seed": tree_seed,
                                "mmd": float(risk.mean()), "variance_mae": float(variance_error.mean()),
                                "bias": float(bias.mean()), "remainder": float((risk - bias).mean()),
                                "n_eff": float((1.0 / np.sum(weights**2, axis=1)).mean()),
                                "max_prediction_mismatch": mismatch})
            write_csv(args.output / "drf_smooth_replicates.csv", records)
            print(f"eps={eps}: paired repetition {rep+1}/{args.reps}", flush=True)

    summaries, paired = [], []
    for eps in EPSILONS:
        for method in METHODS:
            subset = [r for r in records if r["epsilon"] == eps and r["method"] == method]
            row = {"epsilon": eps, "method": method, "reps": len(subset)}
            for metric in METRICS:
                row[metric+"_mean"], row[metric+"_mcse"] = mean_mcse([r[metric] for r in subset])
            summaries.append(row)
        reference = {r["rep"]: r for r in records if r["epsilon"] == eps and r["method"] == METHODS[0]}
        for metric in METRICS:
            difference = [r[metric] - reference[r["rep"]][metric]
                          for r in records if r["epsilon"] == eps and r["method"] == METHODS[1]]
            mean, mcse = mean_mcse(difference)
            paired.append({"epsilon": eps, "metric": metric, "comparison": "RFF-MMD minus Mean-CART",
                           "mean_difference": mean, "paired_mcse": mcse, "reps": len(difference)})
    write_csv(args.output / "drf_smooth_summary.csv", summaries)
    write_csv(args.output / "drf_smooth_paired.csv", paired)

    report_style()
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH_IN, 2.8))
    t = np.linspace(0, 1, 400)
    for eps in EPSILONS:
        axes[0].plot(t, variance_function(t, eps), label="step" if eps == 0 else rf"$\varepsilon={eps:g}$")
    axes[0].set(xlabel=r"$x^{(1)}$", title=r"$\mathrm{Var}(Y\mid X=x)$")
    axes[0].legend(loc="lower right", fontsize=8, handlelength=1.2, labelspacing=0.25, borderaxespad=0.3)
    positions = np.arange(len(EPSILONS))
    for j, method in enumerate(METHODS):
        rows = [next(r for r in summaries if r["epsilon"] == e and r["method"] == method) for e in EPSILONS]
        for ax, metric in zip(axes[1:], ["mmd", "variance_mae"]):
            ax.errorbar(positions + (j-.5)*.1, [r[metric+"_mean"] for r in rows],
                        yerr=[1.96*r[metric+"_mcse"] for r in rows], marker="o", markersize=4,
                        capsize=2.5, label=method)
    for ax, title in zip(axes[1:], [r"Conditional MMD$^2$", "Conditional-variance MAE"]):
        ax.set_xticks(positions, ["step"] + [f"{e:g}" for e in EPSILONS[1:]])
        ax.set(xlabel=r"transition width $\varepsilon$", title=title)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.67, 0))
    fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=0.6)
    fig.savefig(args.output / "drf_smooth_transition.pdf")
    plt.close(fig)
    write_metadata(args.output / "smooth_metadata.json", args, [__file__, Path(__file__).with_name("drf_core.py")])
    print("Smooth-transition experiment saved.", flush=True)


if __name__ == "__main__":
    main()
