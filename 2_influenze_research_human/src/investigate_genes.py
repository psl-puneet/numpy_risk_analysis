
"""Inspect selected antiviral/interferon genes after probe annotation."""
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"

# Examples of commonly studied interferon/antiviral response genes.
# This is a hypothesis-driven list, not a claim that they are significant here.
TARGET_GENES = {
    "IFIT1", "IFIT2", "IFIT3", "ISG15", "MX1", "MX2",
    "OAS1", "OAS2", "OAS3", "IFIH1", "DDX58", "IRF7",
    "STAT1", "CXCL10", "IFNB1", "IFITM3", "RSAD2",
}


def main():
    annotated_path = RESULTS_DIR / "annotated_probe_differences.csv"
    if not annotated_path.exists():
        raise FileNotFoundError("Run `python src/annotate_probes.py` first.")

    df = pd.read_csv(annotated_path)
    symbol_col = next(
        (c for c in df.columns if c.lower() in {"gene_symbol", "symbol", "gene symbols"}),
        None,
    )
    if symbol_col is None:
        raise RuntimeError(
            "No gene-symbol column found. Inspect the platform annotation columns "
            "and update annotate_probes.py to map the correct field."
        )

    df["gene_symbol_clean"] = (
        df[symbol_col].fillna("").astype(str).str.upper()
        .str.replace(r"\s+", "", regex=True)
    )
    df["is_target_gene"] = df["gene_symbol_clean"].apply(
        lambda value: any(gene in value.replace(";", ",").split(",") for gene in TARGET_GENES)
    )
    selected = df[df["is_target_gene"]].copy()
    selected = selected.sort_values(
        ["adjusted_p_value", "absolute_difference"],
        ascending=[True, False],
        na_position="last",
    )
    output = RESULTS_DIR / "antiviral_gene_results.csv"
    selected.to_csv(output, index=False)

    print(selected.to_string(index=False))
    print(f"\nSaved: {output}")
    print("Inspect effect sizes and adjusted p-values together; do not infer significance from fold/change alone.")


if __name__ == "__main__":
    main()
