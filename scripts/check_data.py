"""Check that the case data is in data/raw/ and print its shape.

The data is synthetic and was shared with Mastercard Data Quest 2026
participants; it is not redistributed in this repository.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hidden_biz import config  # noqa: E402

if __name__ == "__main__":
    missing = [p for p in (config.BUSINESS_FILE, config.CONSUMER_FILE, config.MERCHANTS_FILE) if not p.exists()]
    if missing:
        sys.exit("Missing: " + ", ".join(p.name for p in missing) + f"\nPut the case files into {config.DATA_RAW}")
    for path in (config.BUSINESS_FILE, config.CONSUMER_FILE, config.MERCHANTS_FILE):
        meta = pd.read_parquet(path, columns=[]).shape
        print(f"{path.name}: {meta[0]:,} rows")
