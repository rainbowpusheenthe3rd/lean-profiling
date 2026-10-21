"""Alerts on problematic patterns in the data (port of ydata-profiling's ``model/alerts.py``)."""

import zlib
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

ALERT_STYLES = {
    "constant": "warning",
    "constant_length": "primary",
    "duplicates": "secondary",
    "empty": "info",
    "high_cardinality": "danger",
    "high_correlation": "secondary",
    "infinite": "info",
    "imbalance": "primary",
    "missing": "info",
    "non_stationary": "secondary",
    "rejected": "danger",
    "seasonal": "secondary",
    "skewed": "info",
    "truncated": "info",
    "type_date": "warning",
    "uniform": "danger",
    "unique": "danger",
    "unsupported": "warning",
    "zeros": "info",
}

# Enum order in ydata-profiling; alerts are sorted by "AlertType.<NAME>".
FIELDS: Dict[str, Set[str]] = {
    "CONSTANT": {"n_distinct"},
    "ZEROS": {"n_zeros", "p_zeros"},
    "HIGH_CORRELATION": set(),
    "HIGH_CARDINALITY": {"n_distinct"},
    "UNSUPPORTED": set(),
    "DUPLICATES": {"n_duplicates"},
    "SKEWED": {"skewness"},
    "IMBALANCE": {"imbalance"},
    "MISSING": {"p_missing", "n_missing"},
    "INFINITE": {"p_infinite", "n_infinite"},
    "TYPE_DATE": set(),
    "UNIQUE": {"n_distinct", "p_distinct", "n_unique", "p_unique"},
    "CONSTANT_LENGTH": {"composition_min_length", "composition_max_length"},
    "REJECTED": set(),
    "UNIFORM": set(),
    "EMPTY": {"n"},
}


def var_id(name: str) -> int:
    """Stable anchor id for a variable (ydata-profiling used Python's per-process ``hash``)."""
    return zlib.crc32(str(name).encode("utf-8"))


def fmt_percent(value: float, edge_cases: bool = True) -> str:
    if edge_cases and round(value, 3) == 0 and value > 0:
        return "< 0.1%"
    if edge_cases and round(value, 3) == 1 and value < 1:
        return "> 99.9%"
    return f"{value * 100:2.1f}%"


class Alert:
    def __init__(self, name: str, values: Optional[dict] = None, column_name: Optional[str] = None):
        self.name = name
        self.values = values or {}
        self.column_name = column_name
        self.fields = FIELDS[name]

    @property
    def anchor_id(self) -> int:
        return var_id(self.column_name)

    @property
    def style(self) -> str:
        return ALERT_STYLES.get(self.name.lower(), "secondary")

    @property
    def type_name(self) -> str:
        return self.name.replace("_", " ").capitalize()

    def __repr__(self) -> str:
        v, col = self.values, self.column_name
        descriptions = {
            "CONSTANT": lambda: f"[{col}] has a constant value",
            "CONSTANT_LENGTH": lambda: f"[{col}] has a constant length",
            "DUPLICATES": lambda: f"Dataset has {v['n_duplicates']} ({fmt_percent(v['p_duplicates'])}) duplicate rows",
            "EMPTY": lambda: "Dataset is empty",
            "HIGH_CARDINALITY": lambda: f"[{col}] has {v['n_distinct']} ({fmt_percent(v['p_distinct'])}) distinct values",
            "HIGH_CORRELATION": lambda: f"[{col}] is highly {v['corr']} correlated with [{v['fields'][0]}]"
            + (f" and {len(v['fields']) - 1} other fields" if len(v["fields"]) > 1 else ""),
            "IMBALANCE": lambda: f"[{col}] is highly imbalanced ({v['imbalance']})",
            "INFINITE": lambda: f"[{col}] has {v['n_infinite']} ({fmt_percent(v['p_infinite'])}) infinite values",
            "MISSING": lambda: f"[{col}] {v['n_missing']} ({fmt_percent(v['p_missing'])}) missing values",
            "SKEWED": lambda: f"[{col}] is highly skewed(γ1 = {v['skewness']})",
            "TYPE_DATE": lambda: f"[{col}] only contains datetime values, but is categorical. "
            "Consider applying `pd.to_datetime()`",
            "UNIFORM": lambda: f"[{col}] is uniformly distributed",
            "UNIQUE": lambda: f"[{col}] has unique values",
            "UNSUPPORTED": lambda: f"[{col}] is an unsupported type, check if it needs cleaning or further analysis",
            "ZEROS": lambda: f"[{col}] has {v['n_zeros']} ({fmt_percent(v['p_zeros'])}) zeros",
            "REJECTED": lambda: f"[{col}] was rejected",
        }
        return descriptions[self.name]()


def _alert_value(value: float) -> bool:
    return not pd.isna(value) and value > 0.01


def _variable_alerts(config, description: dict) -> List[str]:
    names: List[str] = []
    if _alert_value(description["p_missing"]):
        names.append("MISSING")
    vtype = description["type"]
    if vtype == "Unsupported":
        return names + ["UNSUPPORTED", "REJECTED"]
    if description.get("n_distinct", np.nan) == description["n"]:
        names.append("UNIQUE")
    if description.get("n_distinct", np.nan) == 1:
        names.append("CONSTANT")
    if vtype == "Categorical":
        if description.get("n_distinct", np.nan) > config.vars.cat.cardinality_threshold:
            names.append("HIGH_CARDINALITY")
        if "chi_squared" in description and description["chi_squared"]["pvalue"] > config.vars.cat.chi_squared_threshold:
            names.append("UNIFORM")
        if "imbalance" in description and description["imbalance"] > config.vars.cat.imbalance_threshold:
            names.append("IMBALANCE")
    if vtype == "Numeric":
        threshold = config.vars.num.skewness_threshold
        skew = description["skewness"]
        if not pd.isna(skew) and (skew < -threshold or skew > threshold):
            names.append("SKEWED")
        if _alert_value(description["p_infinite"]):
            names.append("INFINITE")
        if _alert_value(description["p_zeros"]):
            names.append("ZEROS")
        if "chi_squared" in description and description["chi_squared"]["pvalue"] > config.vars.num.chi_squared_threshold:
            names.append("UNIFORM")
    if vtype == "Boolean":
        if "imbalance" in description and description["imbalance"] > config.vars.bool.imbalance_threshold:
            names.append("IMBALANCE")
    return names


def get_alerts(config, table: dict, variables: dict, correlations: dict) -> List[Alert]:
    alerts: List[Alert] = []
    if _alert_value(table.get("n_duplicates", np.nan)):
        alerts.append(Alert("DUPLICATES", values=table))
    if table["n"] == 0:
        alerts.append(Alert("EMPTY", values=table))
    for col, description in variables.items():
        alerts += [Alert(name, values=description, column_name=col) for name in _variable_alerts(config, description)]

    from lean_profiling.describe import check_correlation

    consolidated: Dict[str, List[str]] = {}
    for corr, matrix in correlations.items():
        settings = config.correlations[corr]
        if settings.get("warn_high_correlations", True):
            consolidated.update(check_correlation(matrix, settings.get("threshold", 0.9)))
    for col, fields in consolidated.items():
        alerts.append(Alert("HIGH_CORRELATION", values={"corr": "overall", "fields": fields}, column_name=col))

    alerts.sort(key=lambda alert: f"AlertType.{alert.name}")
    return alerts
