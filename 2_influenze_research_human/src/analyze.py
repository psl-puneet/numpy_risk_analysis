
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
