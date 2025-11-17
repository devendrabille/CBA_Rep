import os
from typing import List, Dict, Union
import pandas as pd

__all__ = ["load_data", "load_csvs"]

def _read_csv_like(src) -> pd.DataFrame:
    """Read a CSV from a path-like or a file-like (Streamlit uploaded file)."""
    # Streamlit uploaded files have a .read / .name attribute
    if hasattr(src, "read"):
        # file-like object
        src.seek(0)
        return pd.read_csv(src)
    # path-like / string path
    return pd.read_csv(str(src))

def load_data(src: Union[str, os.PathLike, object]) -> pd.DataFrame:
    """
    Backwards-compatible single-file loader.
    src can be a filesystem path (str/Path) or a Streamlit UploadedFile.
    """
    try:
        return _read_csv_like(src)
    except Exception as e:
        raise RuntimeError(f"Failed to load CSV from {getattr(src, 'name', str(src))}: {e}") from e

def load_csvs(uploaded_files: List[object]) -> Dict[str, pd.DataFrame]:
    """
    Load multiple CSVs. Accepts a list of Streamlit UploadedFile objects or paths.
    Returns dict mapping filename -> DataFrame (None if load failed).
    """
    dfs: Dict[str, pd.DataFrame] = {}
    for f in uploaded_files:
        name = getattr(f, "name", None) or os.path.basename(str(f))
        try:
            dfs[name] = _read_csv_like(f)
        except Exception:
            dfs[name] = None
    return dfs