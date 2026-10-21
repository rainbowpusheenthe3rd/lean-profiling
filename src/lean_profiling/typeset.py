"""Variable type inference.

A port of ydata-profiling's visions typeset. Detection walks from ``Unsupported``
to the type whose dtype check matches, then inference tries to convert text
columns. ydata-profiling visits candidate types in set order, which varies
between runs for ambiguous columns (e.g. a few repeated date strings can come
out as DateTime or Categorical); here the order is fixed:
Boolean, Numeric, DateTime, Categorical.
"""

import datetime
from typing import Callable, Tuple

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from lean_profiling.config import Settings


def _non_null(fn: Callable[..., bool]) -> Callable[..., bool]:
    def inner(series: pd.Series, *args) -> bool:
        if series.hasnans:
            series = series.dropna()
            if series.empty:
                return False
        return fn(series, *args)

    return inner


def _safe(fn: Callable[..., bool]) -> Callable[..., bool]:
    def inner(*args) -> bool:
        try:
            return bool(fn(*args))
        except Exception:  # noqa: BLE001 - mirrors ydata's bare except
            return False

    return inner


# Dtype checks (visions "contains" operations).


@_non_null
def is_numeric(series: pd.Series) -> bool:
    return pdt.is_numeric_dtype(series) and not pdt.is_bool_dtype(series)


@_non_null
def _series_is_string(series: pd.Series) -> bool:
    if not all(isinstance(v, str) for v in series.values[0:5]):
        return False
    try:
        return bool((series.astype(str).values == series.values).all())
    except (TypeError, ValueError):
        return False


@_non_null
def is_text(series: pd.Series) -> bool:
    return (
        not isinstance(series.dtype, pd.CategoricalDtype)
        and pdt.is_string_dtype(series)
        and _series_is_string(series)
    )


@_non_null
def is_datetime(series: pd.Series) -> bool:
    if pdt.is_datetime64_any_dtype(series):
        return True
    return bool(series.apply(type).isin([datetime.date, datetime.datetime]).all())


@_non_null
def is_categorical(series: pd.Series) -> bool:
    return isinstance(series.dtype, pd.CategoricalDtype) and not pdt.is_bool_dtype(series)


@_non_null
def is_boolean(series: pd.Series) -> bool:
    if pdt.is_object_dtype(series):
        try:
            return bool(series.isin({True, False}).all())
        except Exception:  # noqa: BLE001
            return False
    return pdt.is_bool_dtype(series)


# Inference relations (convert text into a more specific type).


def _string_is_bool(series: pd.Series, mapping: dict) -> bool:
    if isinstance(series.dtype, pd.CategoricalDtype):
        return False
    return _non_null(_safe(lambda s: s.str.lower().isin(mapping.keys()).all()))(series)


def _numeric_is_category(series: pd.Series, config: Settings) -> bool:
    return 1 <= series.nunique() <= config.vars.num.low_categorical_threshold


@_non_null
def _object_is_bool(series: pd.Series) -> bool:
    if pdt.is_object_dtype(series):
        try:
            return all(item in {True, False} for item in series)
        except Exception:  # noqa: BLE001
            return False
    return False


@_non_null
def _string_is_numeric(series: pd.Series, config: Settings) -> bool:
    if pdt.is_bool_dtype(series) or _object_is_bool(series):
        return False
    try:
        _ = series.astype(float)
        converted = pd.to_numeric(series, errors="coerce")
        if converted.hasnans and converted.count() == 0:
            return False
    except Exception:  # noqa: BLE001
        return False
    return not _numeric_is_category(series, config)


def _to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="mixed")


@_non_null
def _string_is_datetime(series: pd.Series) -> bool:
    try:
        return not _to_datetime(series).isna().all()
    except Exception:  # noqa: BLE001
        return False


@_non_null
def _string_is_category(series: pd.Series, config: Settings) -> bool:
    n_unique = series.nunique()
    unique_threshold = config.vars.cat.percentage_cat_threshold
    ratio = n_unique / series.size
    ratio_ok = ratio < unique_threshold if unique_threshold <= 1 else ratio <= unique_threshold
    return (
        1 <= n_unique <= config.vars.cat.cardinality_threshold
        and ratio_ok
        and not _string_is_bool(series, config.vars.bool.mappings)
    )


def _to_category(series: pd.Series) -> pd.Series:
    hasnans = series.hasnans
    values = series.astype(str)
    if hasnans:
        values = values.replace("nan", np.nan).replace("<NA>", np.nan)
    return values.astype("string")


def _to_bool(series: pd.Series, mapping: dict) -> pd.Series:
    mapped = series.str.lower().map(mapping)
    return mapped.astype("boolean" if mapped.hasnans else bool)


def infer(series: pd.Series, config: Settings) -> Tuple[str, pd.Series]:
    """Return the inferred type name and the series cast to that type."""
    for name, check in (
        ("Boolean", is_boolean),
        ("Numeric", is_numeric),
        ("Text", is_text),
        ("Categorical", is_categorical),
        ("DateTime", is_datetime),
    ):
        if not series.empty and check(series):
            vtype = name
            break
    else:
        return "Unsupported", series

    if not config.infer_dtypes:
        return vtype, series

    if vtype == "Text":
        mapping = config.vars.bool.mappings
        if _string_is_bool(series, mapping):
            return "Boolean", _to_bool(series, mapping)
        if _string_is_numeric(series, config):
            series = pd.to_numeric(series, errors="coerce")
            vtype = "Numeric"
        elif _string_is_datetime(series):
            return "DateTime", _to_datetime(series)
        elif _string_is_category(series, config):
            return "Categorical", _to_category(series)
        else:
            return "Text", series

    if vtype == "Numeric" and _numeric_is_category(series, config):
        return "Categorical", _to_category(series)
    return vtype, series
