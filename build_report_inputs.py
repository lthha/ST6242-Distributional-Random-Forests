"""Generate LaTeX table rows and numeric macros directly from experiment CSVs."""
import argparse
import csv
from pathlib import Path
import shutil


def read(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def value(row, field, digits=5):
    return f"{float(row[field]):.{digits}f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--report", type=Path, default=Path("report"))
    parser.add_argument("--also-copy-to", type=Path)
    args = parser.parse_args()
    args.report.mkdir(parents=True, exist_ok=True)
    root = read(args.results / "drf_main_summary.csv")
    bandwidth = read(args.results / "drf_bandwidth_summary.csv")
    forest = read(args.results / "drf_full_forest_summary.csv")
    paired = read(args.results / "drf_full_forest_paired.csv")
    root_paired = read(args.results / "drf_main_paired.csv")
    bandwidth_paired = read(args.results / "drf_bandwidth_paired.csv")
    quadrature = read(args.results / "drf_quadrature_check.csv")
    labels = {"variance": "Variance change", "tail": "Tail-shape change",
              "bimodality": "Matched-moment bimodality", "dependence": "Dependence change"}
    root_labels = {"CART on Y": r"CART on $Y$", "CART on Y^2": r"CART on $Y^2$", "MMD (h=1)": r"MMD ($h=1$)"}
    outputs = {}
    lines = []
    for row in root:
        lines.append(f"{row['n']} & {root_labels[row['method']]} & {value(row, 'mean_abs_error')} & {value(row, 'mcse')} & {value(row, 'median_abs_error')} & {value(row, 'within_0.1', 3)} " + r"\\")
    outputs["root_rows.tex"] = "\n".join(lines)
    outputs["bandwidth_rows.tex"] = "\n".join(f"{float(r['h']):.2f} & {value(r,'mean_abs_error')} & {value(r,'mcse')} & {value(r,'median_abs_error')} & {value(r,'within_0.1',3)} " + r"\\" for r in bandwidth)
    for metric in ["mmd", "functional"]:
        lines = []
        for scenario, label in labels.items():
            a, b = [next(r for r in forest if r["scenario"] == scenario and r["method"] == m) for m in ["Mean-CART", "RFF-MMD"]]
            lines.append(f"{label} & ${value(a,metric+'_mean')}\\pm {value(a,metric+'_mcse')}$ & ${value(b,metric+'_mean')}\\pm {value(b,metric+'_mcse')}$ " + r"\\")
        outputs[f"forest_{metric}_rows.tex"] = "\n".join(lines)
    lines = []
    for scenario, label in labels.items():
        mmd, functional = [next(r for r in paired if r["scenario"] == scenario and r["metric"] == m) for m in ["mmd", "functional"]]
        lines.append(f"{label} & ${value(mmd,'mean_difference')}\\pm {value(mmd,'paired_mcse')}$ & ${value(functional,'mean_difference')}\\pm {value(functional,'paired_mcse')}$ " + r"\\")
    outputs["forest_paired_rows.tex"] = "\n".join(lines)
    macros = {}
    for name, method in [("RootMMDMAE", "MMD (h=1)"), ("RootSquareMAE", "CART on Y^2")]:
        row = next(r for r in root if r["n"] == "800" and r["method"] == method)
        macros[name] = value(row, "mean_abs_error", 4)
    row = next(r for r in root_paired if r["n"] == "800" and r["comparison"].startswith("MMD"))
    macros["RootPairedDifference"], macros["RootPairedMCSE"] = value(row,"mean_difference"), value(row,"paired_mcse")
    best = min(bandwidth, key=lambda r: float(r["mean_abs_error"]))
    macros["BestBandwidth"], macros["BestBandwidthMAE"] = f"{float(best['h']):g}", value(best,"mean_abs_error")
    row = next(r for r in bandwidth if float(r["h"]) == 1)
    macros["BandwidthOneMAE"] = value(row,"mean_abs_error")
    row = next(r for r in bandwidth_paired if float(r["h"]) == 2)
    macros["BandwidthPairedDifference"], macros["BandwidthPairedMCSE"] = value(row,"mean_difference"), value(row,"paired_mcse")
    for name, scenario, method in [("CovarianceMeanMAE","dependence","Mean-CART"), ("CovarianceRFFMAE","dependence","RFF-MMD"), ("TailMeanMAE","tail","Mean-CART"), ("TailRFFMAE","tail","RFF-MMD")]:
        row = next(r for r in forest if r["scenario"] == scenario and r["method"] == method)
        macros[name] = value(row,"functional_mean",4)
    for name, field in [("QuadratureMeanChange","absolute_mean_change"), ("QuadratureMaxChange","max_query_change")]:
        number = max(float(r[field]) for r in quadrature if r["quadrature"] == "4096")
        mantissa, exponent = f"{number:.2e}".split("e")
        macros[name] = f"{mantissa}\\times10^{{{int(exponent)}}}"
    macros["MMDMeanWins"] = str(sum(float(r["mean_difference"]) > 0 for r in paired if r["metric"] == "mmd"))
    macros["MMDRFFWins"] = str(sum(float(r["mean_difference"]) < 0 for r in paired if r["metric"] == "mmd"))
    macros["FunctionalRFFWins"] = str(sum(float(r["mean_difference"]) < 0 for r in paired if r["metric"] == "functional"))
    outputs["experiment_numbers.tex"] = "\n".join("\\newcommand{\\"+k+"}{"+v+"}" for k,v in macros.items())
    # Include complete tabular environments so LaTeX file hooks cannot interfere
    # with booktabs' alignment commands between rows.
    tables = {
        "root_rows.tex": ("rlrrrr", r"$n$ & Split rule & Mean abs. error & MCSE & Median abs. error & Within $0.1$"),
        "bandwidth_rows.tex": ("rrrrr", r"$h$ & Mean abs. error & MCSE & Median abs. error & Within $0.1$"),
        "forest_mmd_rows.tex": ("lcc", "Scenario & Mean-CART forest & RFF--MMD forest"),
        "forest_functional_rows.tex": ("lcc", "Scenario & Mean-CART forest & RFF--MMD forest"),
        "forest_paired_rows.tex": ("lcc", r"Scenario & Conditional MMD$^2$ difference & Functional MAE difference"),
    }
    for name, (columns, header) in tables.items():
        outputs[name] = ("\\begin{tabular}{@{}" + columns + "@{}}\n\\toprule\n" + header +
                         " \\\\\n\\midrule\n" + outputs[name] + "\n\\bottomrule\n\\end{tabular}")
    for name, content in outputs.items():
        path = args.report / name
        path.write_text("% Generated from CSV records; edit the experiment, not this file.\n" + content + "\n")
        if args.also_copy_to:
            args.also_copy_to.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, args.also_copy_to / name)
    for name in ["drf_population_scores.pdf", "drf_split_boxplots.pdf", "drf_bandwidth_sensitivity.pdf", "drf_full_forest_scores.pdf"]:
        shutil.copy2(args.results/name, args.report/name)
        if args.also_copy_to:
            shutil.copy2(args.results/name, args.also_copy_to/name)
    print("Generated report inputs:", ", ".join(outputs))


if __name__ == "__main__":
    main()
