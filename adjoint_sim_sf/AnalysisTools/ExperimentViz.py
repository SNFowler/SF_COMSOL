"""
ExperimentViz.py
"""

import pandas as pd
import json

def jsonl_to_df(path: str) -> pd.DataFrame:
    """Load a JSONL file into a pandas DataFrame."""
    with open(path, "r") as f:
        data = [json.loads(line) for line in f if line.strip()]
    return pd.DataFrame(data)