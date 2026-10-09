import re
from pathlib import Path

import GEOparse
import numpy as np
import pandas as pd

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Download and parse the GEO series
gse = GEOparse.get_GEO(
    "GSE68849",
    destdir=str(DATA_DIR),
)

print("Number of samples:", len(gse.gsms))

expression_columns = []
sample_records = []

for gsm_id, gsm in gse.gsms.items():
    title = gsm.metadata["title"][0]
    table = gsm.table[["ID_REF", "VALUE"]].copy()

    print(gsm_id, repr(title))

    table = table.drop_duplicates("ID_REF")
    table = table.set_index("ID_REF")
    table = table.rename(columns={"VALUE": title})

    expression_columns.append(table)

    sample_records.append({
        "sample_id": gsm_id,
        "title": title,
    })

expression = pd.concat(
    expression_columns,
    axis=1,
    join="inner",
)

metadata = pd.DataFrame(sample_records)

print("Expression matrix:", expression.shape)
print(metadata.to_string(index=False))
print(expression.iloc[:5, :5])

def classify_sample(title):
    title = str(title).lower()

    match = re.search(r"donor\s+(\d+)", title)
    if not match:
        raise ValueError(f"Cannot identify donor: {title}")

    donor = int(match.group(1))

    if "no virus control" in title:
        condition = "control"
    elif "influenza treated" in title:
        condition = "treated"
    else:
        raise ValueError(f"Unknown condition: {title}")

    return donor, condition


records = []

for title in expression_columns:
    donor, condition = classify_sample(title)
    records.append({
        "sample": title,
        "donor": donor,
        "condition": condition,
    })

meta = pd.DataFrame(records).set_index("sample")

# Ensure each donor has both conditions
control_samples = []
treated_samples = []

for donor in sorted(meta["donor"].unique()):
    donor_meta = meta[meta["donor"] == donor]

    control = donor_meta[
        donor_meta["condition"] == "control"
    ].index

    treated = donor_meta[
        donor_meta["condition"] == "treated"
    ].index

    if len(control) != 1 or len(treated) != 1:
        raise ValueError(f"Unexpected sample pairing for donor {donor}")

    control_samples.append(control[0])
    treated_samples.append(treated[0])

control = expression[control_samples].to_numpy(dtype=float)
treated = expression[treated_samples].to_numpy(dtype=float)

difference = treated - control

mean_difference = difference.mean(axis=1)

# Paired t-test across the five donors
test = ttest_rel(
    treated,
    control,
    axis=1,
    nan_policy="omit",
)

results = pd.DataFrame({
    "probe_id": expression.index,
    "control_mean": control.mean(axis=1),
    "treated_mean": treated.mean(axis=1),
    "mean_difference": mean_difference,
    "p_value": test.pvalue,
})

results["absolute_difference"] = (
    results["mean_difference"].abs()
)

results = results.sort_values(
    "absolute_difference",
    ascending=False,
)

Path("results").mkdir(exist_ok=True)

results.to_csv(
    "results/probe_differences.csv",
    index=False,
)

print(results.head(20).to_string(index=False))
