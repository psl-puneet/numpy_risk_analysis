import re
from pathlib import Path

import GEOparse
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def classify_sample(title: str):
    title_lower = str(title).lower()
    match = re.search(r"donor\s+(\d+)", title_lower)
    if not match:
        raise ValueError(f"Cannot identify donor from sample title: {title}")

    if "no virus control" in title_lower:
        condition = "control"
    elif "influenza treated" in title_lower:
        condition = "treated"
    else:
        raise ValueError(f"Unknown condition in sample title: {title}")

    return int(match.group(1)), condition


def main():
    geo_file = DATA_DIR / "GSE68849_family.soft.gz"
    if geo_file.exists():
        gse = GEOparse.get_GEO(filepath=str(geo_file))
    else:
        gse = GEOparse.get_GEO("GSE68849", destdir=str(DATA_DIR))

    expression_columns = []
    records = []

    for gsm_id, gsm in gse.gsms.items():
        title = gsm.metadata["title"][0]
        table = gsm.table[["ID_REF", "VALUE"]].copy()
        table = table.drop_duplicates("ID_REF").set_index("ID_REF")
        table["VALUE"] = pd.to_numeric(table["VALUE"], errors="coerce")
        table = table.rename(columns={"VALUE": title})
        expression_columns.append(table)

        donor, condition = classify_sample(title)
        records.append({
            "sample_id": gsm_id,
            "sample": title,
            "donor": donor,
            "condition": condition,
        })

    expression = pd.concat(expression_columns, axis=1, join="inner")
    expression.index.name = "probe_id"
    metadata = pd.DataFrame(records)

    # Ensure there is exactly one control and one treated sample per donor.
    counts = metadata.groupby(["donor", "condition"]).size()
    if not (counts == 1).all():
        raise ValueError(f"Unexpected donor/condition pairing:\n{counts}")

    expression.to_csv(RESULTS_DIR / "expression_matrix.csv")
    metadata.to_csv(RESULTS_DIR / "sample_metadata.csv", index=False)

    print(f"Samples: {len(metadata)}")
    print(f"Expression matrix (probes x samples): {expression.shape}")
    print(f"Saved files in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
