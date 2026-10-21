"""Report settings.

Defaults mirror the settings ydata-profiling 4.18's ``ProfileReport`` actually runs
with (its pydantic defaults, which differ from its ``config_default.yaml`` in places,
e.g. 8-digit precision, 10 extreme values, a 0.5 correlation alert threshold). Override any value with keyword arguments to ``ProfileReport``,
using nested dicts for nested sections, e.g. ``ProfileReport(df, plot={"histogram": {"bins": 20}})``.
"""

import copy
import json
from typing import Any, Dict

DEFAULTS: Dict[str, Any] = {
    "title": "Profiling Report",
    "dataset": {
        "description": "",
        "creator": "",
        "author": "",
        "copyright_holder": "",
        "copyright_year": "",
        "url": "",
    },
    "variables": {"descriptions": {}},
    "infer_dtypes": True,
    "show_variable_description": True,
    "vars": {
        "num": {
            "quantiles": [0.05, 0.25, 0.5, 0.75, 0.95],
            "skewness_threshold": 20,
            "low_categorical_threshold": 5,
            "chi_squared_threshold": 0.999,
        },
        "text": {"length": True, "words": True, "characters": True},
        "cat": {
            "length": True,
            "characters": True,
            "words": True,
            "cardinality_threshold": 50,
            "percentage_cat_threshold": 0.5,
            "imbalance_threshold": 0.5,
            "n_obs": 5,
            "chi_squared_threshold": 0.999,
            "stop_words": [],
        },
        "bool": {
            "n_obs": 3,
            "imbalance_threshold": 0.5,
            "mappings": {
                "t": True,
                "f": False,
                "yes": True,
                "no": False,
                "y": True,
                "n": False,
                "true": True,
                "false": False,
            },
        },
    },
    "sort": None,
    "missing_diagrams": {"bar": True, "matrix": True, "heatmap": True},
    "correlation_table": True,
    "correlations": {
        "auto": {"calculate": True, "warn_high_correlations": True, "threshold": 0.5, "n_bins": 10},
    },
    "interactions": {"continuous": True, "targets": []},
    "categorical_maximum_correlation_distinct": 100,
    "memory_deep": False,
    "plot": {
        "missing": {"force_labels": True},
        "scatter_threshold": 1000,
        "histogram": {"bins": 50, "max_bins": 250, "x_axis_labels": True},
        "cat_freq": {"show": True, "max_unique": 10, "colors": None},
    },
    "n_obs_unique": 10,
    "n_extreme_obs": 10,
    "n_freq_table_max": 10,
    "duplicates": {"head": 10, "key": "# duplicates"},
    "samples": {"head": 10, "tail": 10},
    "reject_variables": True,
    "report": {"precision": 8},
    "html": {
        "navbar_show": True,
        "full_width": False,
        "style": {"primary_colors": ["#0d6efd", "#dc3545", "#198754"]},
    },
}

# ``minimal=True``: ydata-profiling's ``config_minimal.yaml`` (which, being loaded from
# YAML, also brings that file's precision and extreme-value count).
MINIMAL: Dict[str, Any] = {
    "infer_dtypes": False,
    "vars": {
        "num": {"chi_squared_threshold": 0},
        "cat": {"length": False, "characters": False, "words": False, "chi_squared_threshold": 0},
    },
    "missing_diagrams": {"bar": False, "matrix": False, "heatmap": False},
    "correlations": {"auto": {"calculate": False}},
    "interactions": {"continuous": False},
    "plot": {"cat_freq": {"max_unique": 0}},
    "duplicates": {"head": 0},
    "samples": {"head": 0, "tail": 0},
    "report": {"precision": 10},
    "n_obs_unique": 5,
    "n_extreme_obs": 5,
}


def _merge(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


class Settings(dict):
    """A nested dict with attribute access (``config.vars.num.quantiles``)."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return Settings(value) if isinstance(value, dict) and not isinstance(value, Settings) else value

    @classmethod
    def build(cls, minimal: bool = False, **overrides: Any) -> "Settings":
        config = copy.deepcopy(DEFAULTS)
        if minimal:
            _merge(config, copy.deepcopy(MINIMAL))
        _merge(config, copy.deepcopy(overrides))
        return cls._wrap(config)

    @classmethod
    def _wrap(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return cls({k: cls._wrap(v) for k, v in value.items()})
        return value

    def json(self) -> str:
        return json.dumps(self)
