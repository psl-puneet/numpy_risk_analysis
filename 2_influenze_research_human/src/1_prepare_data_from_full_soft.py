
from pathlib import Path
from io import StringIO

import pandas as pd

# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_DIR / "data" / "GDS6063_full.soft"
OUTPUT_DIR = PROJECT_DIR / "GDS6063" /"results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Sample metadata based on GDS6063
SAMPLES = [
    ("GSM1684095", 1, "Control"),
    ("GSM1684096", 1, "Influenza"),
    ("GSM1684097", 2, "Control"),
    ("GSM1684098", 2, "Influenza"),
    ("GSM1684099", 3, "Control"),
    ("GSM1684100", 3, "Influenza"),
    ("GSM1684101", 4, "Control"),
    ("GSM1684102", 4, "Influenza"),
    ("GSM1684103", 5, "Control"),
    ("GSM1684104", 5, "Influenza"),
]

# --------------------------------------------------
# 1. Locate and read the GDS expression table
# --------------------------------------------------
if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Cannot find {INPUT_FILE}. "
        "Place GDSfull.soft in the project's data/ directory."
    )

text = INPUT_FILE.read_text(encoding="utf-8", errors="replace")
lines = text.splitlines()

table_start = next(
    (i for i, line in enumerate(lines)
     if line.strip() == "!dataset_table_begin"),
    None,
)

if table_start is None:
    raise ValueError("Could not find !dataset_table_begin in the SOFT file.")

table_lines = []

for line in lines[table_start + 1:]:
    if line.startswith("!dataset_table_end"):
        break
    if line.startswith("^"):
        break
    if line.strip():
        table_lines.append(line)

if not table_lines:
    raise ValueError("The dataset table appears to be empty.")

df = pd.read_csv(
    StringIO("\n".join(table_lines)),
    sep="\t",
    dtype=str,
    na_values=["null", "NULL", ""],
    keep_default_na=True,
    low_memory=False,
)

df.columns = df.columns.str.strip()

if "ID_REF" not in df.columns:
    raise ValueError("The table does not contain the expected ID_REF column.")

# --------------------------------------------------
# 2. Identify the 10 sample expression columns
# --------------------------------------------------
sample_ids = [sample[0] for sample in SAMPLES]
missing = [sample for sample in sample_ids if sample not in df.columns]

if missing:
    raise ValueError(
        f"These sample columns are missing from the table: {missing}"
    )

# Convert expression values to numeric.
# Do not apply another normalization transformation here.
expression = df[["ID_REF"] + sample_ids].copy()

for sample in sample_ids:
    expression[sample] = pd.to_numeric(
        expression[sample], errors="coerce"
    )

# Remove probes with no usable expression values in any sample.
expression = expression.dropna(subset=sample_ids, how="all")

# Keep probe IDs as identifiers, not numeric measurements.
expression = expression.drop_duplicates(subset=["ID_REF"])

expression.to_csv(
    OUTPUT_DIR / "normalized_counts.csv",
    index=False,
    na_rep="",
)

# --------------------------------------------------
# 3. Create sample_info.csv
# --------------------------------------------------
sample_info = pd.DataFrame(
    SAMPLES,
    columns=["sample_id", "donor", "condition"],
)

sample_info["timepoint_hours"] = 8
sample_info["organism"] = "Homo sapiens"
sample_info["cell_type"] = "Plasmacytoid dendritic cells"

# Explicitly identify the matched design.
sample_info["pair_id"] = sample_info["donor"].map(
    lambda donor: f"donor_{donor}"
)

sample_info.to_csv(
    OUTPUT_DIR / "sample_info.csv",
    index=False,
)

# --------------------------------------------------
# 4. Create gene_data.csv from available annotations
# --------------------------------------------------
sample_columns = set(sample_ids)
annotation_columns = [
    column for column in df.columns
    if column not in sample_columns
]

gene_data = df[annotation_columns].copy()

# Use consistent naming for the probe identifier.
if "ID_REF" in gene_data.columns:
    gene_data = gene_data.rename(columns={"ID_REF": "probe_id"})

gene_data = gene_data.drop_duplicates(subset=["probe_id"])

gene_data.to_csv(
    OUTPUT_DIR / "gene_data.csv",
    index=False,
    na_rep="",
)

# --------------------------------------------------
# 5. Print a validation summary
# --------------------------------------------------
print("Data preparation complete.")
print(f"Expression matrix: {expression.shape[0]} probes x "
      f"{len(sample_ids)} samples")
print(f"Sample metadata: {len(sample_info)} samples")
print(f"Probe annotations: {len(gene_data)} probes")

for filename in [
    "normalized_counts.csv",
    "sample_info.csv",
    "gene_data.csv",
]:
    path = OUTPUT_DIR / filename
    print(f"Created: {path} ({path.stat().st_size:,} bytes)")
