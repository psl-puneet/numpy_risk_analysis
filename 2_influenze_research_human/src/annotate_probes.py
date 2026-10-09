"""Annotate probe IDs using the platform annotation table from GEO."""
from pathlib import Path

import GEOparse
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
GSE_ID = "GSE68849"


def find_column(columns, candidates):
    lower_to_original = {str(c).strip().lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    for col in columns:
        col_lower = str(col).strip().lower()
        if any(candidate.lower() in col_lower for candidate in candidates):
            return col
    return None


def main():
    results_path = RESULTS_DIR / "probe_differences.csv"
    if not results_path.exists():
        raise FileNotFoundError("Run `python src/differential_expression.py` first.")

    gse = GEOparse.get_GEO("GSE68849", destdir=str(DATA_DIR))
    if not gse.gpls:
        raise RuntimeError("No platform annotation was found in the GEO series.")

    platform = next(iter(gse.gpls.values()))
    annotation = platform.table.copy()
    print("Platform:", platform.name)
    print("Available annotation columns:")
    print(annotation.columns.tolist())

    id_col = find_column(annotation.columns, ["ID", "ID_REF", "Probe ID"])
    symbol_col = find_column(annotation.columns, ["Symbol", "Gene Symbol", "GENE SYMBOL"])
    title_col = find_column(annotation.columns, ["Gene", "Gene Title", "ENTREZ_GENE_ID", "RefSeq"])

    if id_col is None:
        raise RuntimeError("Could not identify the probe ID column in platform annotation.")

    rename = {id_col: "probe_id"}
    if symbol_col is not None:
        rename[symbol_col] = "gene_symbol"
    if title_col is not None and title_col not in rename:
        rename[title_col] = "gene_annotation"

    annotation = annotation.rename(columns=rename)
    keep = [c for c in ["probe_id", "gene_symbol", "gene_annotation"] if c in annotation.columns]
    annotation = annotation[keep].drop_duplicates("probe_id")

    results = pd.read_csv(results_path)
    merged = results.merge(annotation, on="probe_id", how="left")
    output = RESULTS_DIR / "annotated_probe_differences.csv"
    merged.to_csv(output, index=False)

    print(f"Annotated {merged.get('gene_symbol', pd.Series(dtype=object)).notna().sum()} rows with symbols, if available.")
    print(f"Saved: {output}")
    print("Review the printed annotation columns if gene symbols were not detected; platform schemas vary.")


if __name__ == "__main__":
    main()
