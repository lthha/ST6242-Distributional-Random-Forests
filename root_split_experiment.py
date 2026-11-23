"""Root-split comparison and bandwidth sensitivity; produces records and figures."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from drf_core import (candidate_splits, cart_scores, mmd_split_scores, mean_mcse, write_csv, write_metadata,
                      report_style, TEXT_WIDTH_IN)


def sample(rng, n):
    x = rng.uniform(size=n)
    y = rng.normal(size=n) * np.where(x > 0.5, 2.0, 1.0)
    return x, y


def summarize(records, group_key):
    result = []
    for group in dict.fromkeys(row[group_key] for row in records):
        for method in dict.fromkeys(row["method"] for row in records if row[group_key] == group):
            subset = [r for r in records if r[group_key] == group and r["method"] == method]
            errors = np.array([r["abs_error"] for r in subset])
            mean, mcse = mean_mcse(errors)
            result.append({group_key: group, "method": method, "reps": len(errors),
                           "mean_abs_error": mean, "mcse": mcse,
                           "median_abs_error": float(np.median(errors)),
                           "within_0.1": float(np.mean(errors <= 0.1))})
    return result


def paired_rows(records, group_key, baseline):
    rows = []
    for group in dict.fromkeys(r[group_key] for r in records):
        subset = [r for r in records if r[group_key] == group]
        reference = {r["rep"]: r["abs_error"] for r in subset if r["method"] == baseline}
        for method in dict.fromkeys(r["method"] for r in subset if r["method"] != baseline):
            differences = [r["abs_error"] - reference[r["rep"]] for r in subset if r["method"] == method]
            mean, mcse = mean_mcse(differences)
            rows.append({group_key: group, "comparison": method + " minus " + baseline,
                         "mean_difference": mean, "paired_mcse": mcse, "reps": len(differences)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--bandwidth-reps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=6242)
    parser.add_argument("--bandwidth-seed", type=int, default=6243)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    if min(args.reps, args.bandwidth_reps) < 2:
        parser.error("At least two replications are needed for MCSEs")
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    main_records = []
    for n in [200, 400, 800]:
        for rep in range(args.reps):
            x, y = sample(rng, n)
            order, counts, thresholds = candidate_splits(x)
            scores = {"CART on Y": cart_scores(y[order], counts),
                      "CART on Y^2": cart_scores(y[order]**2, counts),
                      "MMD (h=1)": mmd_split_scores(y[order], counts)}
            for method, score in scores.items():
                threshold = float(thresholds[np.argmax(score)])
                main_records.append({"n": n, "rep": rep, "method": method,
                                     "threshold": threshold, "abs_error": abs(threshold - 0.5)})
        print(f"Completed root splits: n={n}, replications={args.reps}", flush=True)
    write_csv(args.output / "drf_main_splits.csv", main_records)
    write_csv(args.output / "drf_main_summary.csv", summarize(main_records, "n"))
    write_csv(args.output / "drf_main_paired.csv", paired_rows(main_records, "n", "CART on Y^2"))

    # Bandwidth sensitivity. Large bandwidths show the limit of Proposition 2: on data, the MMD split
    # approaches the CART-on-Y split, so both CART rules are evaluated on the same datasets.
    bandwidths = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    rng = np.random.default_rng(args.bandwidth_seed)
    bandwidth_records, baseline_records = [], []
    for rep in range(args.bandwidth_reps):
        x, y = sample(rng, 400)
        order, counts, thresholds = candidate_splits(x)
        cart_index = {"CART on Y": int(np.argmax(cart_scores(y[order], counts))),
                      "CART on Y^2": int(np.argmax(cart_scores(y[order]**2, counts)))}
        for method, index in cart_index.items():
            threshold = float(thresholds[index])
            baseline_records.append({"method": method, "rep": rep,
                                     "threshold": threshold, "abs_error": abs(threshold - 0.5)})
        for h in bandwidths:
            index = int(np.argmax(mmd_split_scores(y[order], counts, h)))
            threshold = float(thresholds[index])
            bandwidth_records.append({"h": h, "rep": rep, "method": f"h={h:g}",
                                      "threshold": threshold, "abs_error": abs(threshold - 0.5),
                                      "same_split_as_cart_y": index == cart_index["CART on Y"],
                                      "same_split_as_cart_y2": index == cart_index["CART on Y^2"]})
    bandwidth_summary = summarize(bandwidth_records, "h")
    for row in bandwidth_summary:
        subset = [r for r in bandwidth_records if r["h"] == row["h"]]
        row["same_split_as_cart_y"] = float(np.mean([r["same_split_as_cart_y"] for r in subset]))
        row["same_split_as_cart_y2"] = float(np.mean([r["same_split_as_cart_y2"] for r in subset]))
    baseline_summary = summarize(baseline_records, "method")
    write_csv(args.output / "drf_bandwidth_splits.csv", bandwidth_records + baseline_records)
    write_csv(args.output / "drf_bandwidth_summary.csv", bandwidth_summary + baseline_summary)
    # Compare each bandwidth with h=1 and with CART on Y^2 on the same datasets.
    paired_bandwidth = []
    references = {"h=1": {r["rep"]: r["abs_error"] for r in bandwidth_records if r["h"] == 1.0},
                  "CART on Y^2": {r["rep"]: r["abs_error"] for r in baseline_records
                                  if r["method"] == "CART on Y^2"}}
    for name, reference in references.items():
        for h in bandwidths:
            if name == "h=1" and h == 1.0:
                continue
            values = [r["abs_error"] - reference[r["rep"]] for r in bandwidth_records if r["h"] == h]
            mean, mcse = mean_mcse(values)
            paired_bandwidth.append({"h": h, "reference": name, "mean_difference": mean, "paired_mcse": mcse})
    write_csv(args.output / "drf_bandwidth_paired.csv", paired_bandwidth)

    report_style()
    t = np.linspace(0.001, 0.999, 999)
    score = np.where(t <= 0.5, t / (1 - t), (1 - t) / t)
    fig, ax = plt.subplots(figsize=(0.79*TEXT_WIDTH_IN, 2.6))
    ax.plot(t, score, label="MMD", linewidth=2)
    ax.plot(t, score, label=r"CART on $Y^2$", linestyle=":", linewidth=2.2)
    ax.plot(t, np.zeros_like(t), label=r"CART on $Y$", linestyle="--")
    ax.axvline(0.5, color="grey", linestyle="--", linewidth=0.8)
    ax.set(xlabel=r"Candidate split $t$", ylabel="Normalized population score", ylim=(-0.03, 1.05))
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(args.output / "drf_population_scores.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH_IN, 2.6), sharey=True)
    for ax, n, letter in zip(axes, [200, 400, 800], "abc"):
        methods = ["CART on Y", "CART on Y^2", "MMD (h=1)"]
        values = [[r["threshold"] for r in main_records if r["n"] == n and r["method"] == m] for m in methods]
        ax.boxplot(values, labels=["CART\non $Y$", "CART\non $Y^2$", "MMD"], showfliers=False)
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
        ax.set(title=f"({letter}) $n={n}$", ylim=(0, 1))
    axes[0].set_ylabel("Estimated root split")
    fig.tight_layout()
    fig.savefig(args.output / "drf_split_boxplots.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(0.77*TEXT_WIDTH_IN, 2.6))
    ax.errorbar([r["h"] for r in bandwidth_summary], [r["mean_abs_error"] for r in bandwidth_summary],
                yerr=[1.96 * r["mcse"] for r in bandwidth_summary], marker="o", capsize=3, label="MMD")
    for row, style in zip(baseline_summary, ["--", ":"]):
        label = r"CART on $Y$" if row["method"] == "CART on Y" else r"CART on $Y^2$"
        ax.axhline(row["mean_abs_error"], color="grey", linestyle=style, linewidth=1.2, label=label)
    ax.set_xscale("log", base=2)
    ax.set_xticks(bandwidths)
    ax.set_xticklabels([f"{h:g}" for h in bandwidths])
    ax.set(xlabel=r"Gaussian bandwidth $h$", ylabel="Mean absolute split error")
    ax.legend(loc="center left")
    fig.tight_layout()
    fig.savefig(args.output / "drf_bandwidth_sensitivity.pdf")
    plt.close(fig)
    write_metadata(args.output / "root_metadata.json", args, [__file__, Path(__file__).with_name("drf_core.py")])
    print("Root-split experiments and figures saved.", flush=True)


if __name__ == "__main__":
    main()
