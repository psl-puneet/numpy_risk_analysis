from pathlib import Path
from urllib.request import urlretrieve
import tarfile

data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

files = {
    "GSE68849_RAW.tar": (
        "https://ftp.ncbi.nlm.nih.gov/geo/series/"
        "GSE68nnn/GSE68849/suppl/GSE68849_RAW.tar"
    ),
    "GSE68849_non-normalized.txt.gz": (
        "https://ftp.ncbi.nlm.nih.gov/geo/series/"
        "GSE68nnn/GSE68849/suppl/GSE68849_non-normalized.txt.gz"
    ),
}

for filename, url in files.items():
    destination = data_dir / filename
    if not destination.exists():
        print("Downloading", filename)
        urlretrieve(url, destination)

archive = data_dir / "GSE68849_RAW.tar"

with tarfile.open(archive, "r:*") as tar:
    print("\nFiles inside raw archive:")
    for member in tar.getmembers():
        print(member.name, member.size, "bytes")