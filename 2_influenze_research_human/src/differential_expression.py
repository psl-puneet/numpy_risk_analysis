"""Exploratory paired differential-expression testing across matched donors."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"


def main():
    matrix_path = RESULTS_DIR / "expression_matrix.csv"
    metadata_path = RESULTS_DIR / "sample_metadata.csv"
    if not matrix_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Run `python src/analyze.py` first.")

    expression = pd.read_csv(matrix_path, index_col="probe_id")
    expression = expression.apply(pd.to_numeric, errors="coerce")
    metadata = pd.read_csv(metadata_path)

    control_samples, treated_samples = [], []
    for donor in sorted(metadata["donor"].unique()):
        group = metadata[metadata["donor"] == donor]
        controls = group.loc[group["condition"] == "control", "sample"].tolist()
        treated = group.loc[group["condition"] == "treated", "sample"].tolist()
        if len(controls) != 1 or len(treated) != 1:
            raise ValueError(f"Donor {donor} must have exactly one control and one treated sample.")
        control_samples.append(controls[0])
        treated_samples.append(treated[0])

    control = expression[control_samples].to_numpy(dtype=float)
    treated = expression[treated_samples].to_numpy(dtype=float)
    complete = np.isfinite(control) & np.isfinite(treated)
    n_pairs = complete.sum(axis=1)

    control_mean = np.nanmean(control, axis=1)
    treated_mean = np.nanmean(treated, axis=1)
    mean_difference = np.nanmean(treated - control, axis=1)
    p_values = np.full(expression.shape[0], np.nan)

    for i in range(expression.shape[0]):
        mask = complete[i]
        if mask.sum() < 3:
            continue
        differences = treated[i, mask] - control[i, mask]
        if np.allclose(differences, differences[0]):
            # Constant differences yield no useful variance estimate.
            p_values[i] = 1.0 if np.allclose(differences, 0) else np.nan
        else:
            p_values[i] = ttest_rel(treated[i, mask], control[i, mask]).pvalue

    results = pd.DataFrame({
        "probe_id": expression.index,
        "control_mean": control_mean,
        "treated_mean": treated_mean,
        "mean_difference": mean_difference,
        "complete_donor_pairs": n_pairs,
        "p_value": p_values,
    })
    results["absolute_difference"] = results["mean_difference"].abs()
    results["adjusted_p_value"] = np.nan

    valid = results["p_value"].notna()
    if valid.any():
        results.loc[valid, "adjusted_p_value"] = multipletests(
            results.loc[valid, "p_value"], method="fdr_bh"
        )[1]

    results = results.sort_values(
        ["adjusted_p_value", "absolute_difference"],
        ascending=[True, False],
        na_position="last",
    )
    output = RESULTS_DIR / "probe_differences.csv"
    results.to_csv(output, index=False)
    print(results.head(20).to_string(index=False))
    print(f"\nSaved: {output}")
    print("Interpret results as exploratory: only five paired donors are available.")


if __name__ == "__main__":
    main()
