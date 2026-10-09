"""Exploratory checks: missingness, distributions, correlations, donor pairing."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"


def main():
    matrix_path = RESULTS_DIR / "expression_matrix.csv"
    metadata_path = RESULTS_DIR / "sample_metadata.csv"
    if not matrix_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Run `python src/analyze.py` first.")

    expression = pd.read_csv(matrix_path, index_col="probe_id")
    metadata = pd.read_csv(metadata_path)
    expression = expression.apply(pd.to_numeric, errors="coerce")

    print("Matrix shape:", expression.shape)
    print("\nMissing values per sample:")
    print(expression.isna().sum().to_string())
    print(f"\nOverall missingness: {expression.isna().mean().mean():.2%}")

    summary = expression.describe().T
    summary["missing_count"] = expression.isna().sum()
    summary.to_csv(RESULTS_DIR / "sample_summary.csv")

    # Boxplot: sample distributions
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=expression, showfliers=False)
    plt.xticks(rotation=90)
    plt.ylabel("Expression value")
    plt.title("Expression distribution by sample")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "expression_boxplot.png", dpi=150)
    plt.close()

    # Correlation between samples
    corr = expression.corr(method="pearson")
    corr.to_csv(RESULTS_DIR / "sample_correlations.csv")
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, vmin=-1, vmax=1, cmap="coolwarm", annot=True, fmt=".2f")
    plt.title("Sample expression correlation")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "sample_correlation_heatmap.png", dpi=150)
    plt.close()

    # Compare each donor's matched control and treated samples.
    rows = []
    for donor, group in metadata.groupby("donor"):
        control_names = group.loc[group["condition"] == "control", "sample"].tolist()
        treated_names = group.loc[group["condition"] == "treated", "sample"].tolist()
        if len(control_names) != 1 or len(treated_names) != 1:
            raise ValueError(f"Donor {donor} does not have exactly one sample per condition.")

        control = expression[control_names[0]]
        treated = expression[treated_names[0]]
        matched = pd.concat([control, treated], axis=1).dropna()
        rows.append({
            "donor": donor,
            "control_sample": control_names[0],
            "treated_sample": treated_names[0],
            "complete_probes": len(matched),
            "control_mean": matched.iloc[:, 0].mean(),
            "treated_mean": matched.iloc[:, 1].mean(),
            "mean_difference": (matched.iloc[:, 1] - matched.iloc[:, 0]).mean(),
            "pearson_correlation": matched.iloc[:, 0].corr(matched.iloc[:, 1]),
        })

    donor_summary = pd.DataFrame(rows)
    donor_summary.to_csv(RESULTS_DIR / "matched_donor_summary.csv", index=False)
    print("\nMatched donor summary:")
    print(donor_summary.to_string(index=False))
    print(f"\nEDA outputs saved in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
