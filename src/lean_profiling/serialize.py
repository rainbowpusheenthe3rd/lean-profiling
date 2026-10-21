"""JSON export with the same layout as ydata-profiling's ``ProfileReport.to_json()``."""

import json
from typing import Any

import numpy as np
import pandas as pd


def _format(value: Any) -> Any:
    """model/summarizer.format_summary: Series become dicts, histograms become counts/bin_edges."""
    if isinstance(value, dict):
        return {k: _format(v) for k, v in value.items()}
    if isinstance(value, pd.Series):
        return _format(value.to_dict())
    if isinstance(value, tuple) and len(value) == 2 and all(isinstance(x, np.ndarray) for x in value):
        return {"counts": value[0].tolist(), "bin_edges": value[1].tolist()}
    return value


def _encode(value: Any) -> Any:
    """profile_report._render_json.encode_it."""
    if isinstance(value, dict):
        return {_encode(k): _encode(v) for k, v in value.items()}
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [_encode(v) for v in value]
    if isinstance(value, pd.Series):
        return _encode(value.to_list())
    if isinstance(value, pd.DataFrame):
        return _encode(value.to_dict(orient="records"))
    if isinstance(value, np.ndarray):
        return _encode(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def to_json(description: dict) -> str:
    return json.dumps(_encode(_format(description)), indent=4)
