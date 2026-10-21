"""Statistics for each variable and for the dataset as a whole.

A port of ydata-profiling's ``model`` package for pandas. Each function notes
the ydata-profiling module it mirrors so results can be checked side by side.
"""

import itertools
import string
import warnings
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from lean_profiling import stats
from lean_profiling.alerts import get_alerts
from lean_profiling.config import Settings
from lean_profiling.typeset import infer

UNKNOWN = "(unknown)"


# --- summary_algorithms.py ---------------------------------------------------


def histogram_compute(
    config: Settings,
    finite_values: np.ndarray,
    n_unique: int,
    name: str = "histogram",
    weights: Optional[np.ndarray] = None,
) -> dict:
    if len(finite_values) == 0:
        return {name: []}
    hist_config = config.plot.histogram
    finite = finite_values[np.isfinite(finite_values)]
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if vmax - vmin == 0:
        eps = 0.5 if vmin == 0 else abs(vmin) * 0.1
        bins = np.array([vmin - eps, vmin + eps])
    elif hist_config.bins > 0:
        n_bins = max(1, min(hist_config.bins, n_unique, hist_config.max_bins))
        bins = np.linspace(vmin, vmax, n_bins + 1)
    else:
        bins = np.histogram_bin_edges(finite_values, bins="auto")
        if len(bins) - 1 > hist_config.max_bins:
            bins = np.linspace(vmin, vmax, hist_config.max_bins + 1)
    return {name: np.histogram(finite_values, bins=bins, weights=weights)}


def chi_square(values: Optional[np.ndarray] = None, histogram: Optional[np.ndarray] = None) -> dict:
    if histogram is None:
        if values is None:
            return {"statistic": 0, "pvalue": 0}
        try:
            bins = np.histogram_bin_edges(values, bins="auto")
        except ValueError:
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                return {"statistic": 0, "pvalue": 0}
            vmin, vmax = float(finite.min()), float(finite.max())
            bins = np.array([vmin - 0.5, vmin + 0.5]) if vmin == vmax else np.array([vmin, vmax])
        histogram, _ = np.histogram(values, bins=bins)
    if histogram.size == 0 or histogram.sum() == 0:
        return {"statistic": 0, "pvalue": 0}
    return stats.chisquare(histogram)


def imbalance_score(value_counts: pd.Series, n_classes: int) -> float:
    if n_classes > 1:
        return 1 - stats.entropy_base2(value_counts.to_numpy(dtype=float)) / np.log2(n_classes)
    return 0


def weighted_median(data: np.ndarray, weights: np.ndarray) -> Any:
    data, weights = np.asarray(data), np.asarray(weights)
    s_data, s_weights = map(np.sort, [data, weights])
    midpoint = 0.5 * np.sum(s_weights)
    if s_weights[-1] > midpoint:
        return data[weights == np.max(weights)][0]
    cs_weights = np.cumsum(s_weights)
    idx = np.where(cs_weights <= midpoint)[0][-1]
    if cs_weights[idx] == midpoint:
        return np.mean(s_data[idx : idx + 2])
    return s_data[idx + 1]


# --- describe_counts / generic / supported ------------------------------------


def describe_counts(series: pd.Series, summary: dict) -> None:
    try:
        vc_with_nan = series.value_counts(dropna=False)
        _ = set(vc_with_nan.index)
        hashable = True
    except Exception:  # noqa: BLE001
        hashable = False
    summary["hashable"] = hashable
    if hashable:
        vc_with_nan = vc_with_nan[vc_with_nan > 0]
        null_index = vc_with_nan.index.isnull()
        if null_index.any():
            n_missing = vc_with_nan[null_index].sum()
            vc = vc_with_nan[~null_index]
        else:
            n_missing = 0
            vc = vc_with_nan
        summary["value_counts_without_nan"] = vc
        try:
            summary["value_counts_index_sorted"] = vc.sort_index(ascending=True)
            ordering = True
        except TypeError:
            ordering = False
    else:
        n_missing = series.isna().sum()
        ordering = False
    summary["ordering"] = ordering
    summary["n_missing"] = n_missing


def describe_generic(config: Settings, series: pd.Series, summary: dict) -> None:
    length = len(series)
    summary.update(
        {
            "n": length,
            "p_missing": summary["n_missing"] / length if length > 0 else 0,
            "count": length - summary["n_missing"],
            "memory_size": series.memory_usage(deep=config.memory_deep),
        }
    )


def describe_supported(summary: dict) -> dict:
    if not summary["hashable"]:
        return summary
    count = summary["count"]
    vc = summary["value_counts_without_nan"]
    distinct = len(vc)
    unique = vc.where(vc == 1).count()
    out = {
        "n_distinct": distinct,
        "p_distinct": distinct / count if count > 0 else 0,
        "is_unique": unique == count and count > 0,
        "n_unique": unique,
        "p_unique": unique / count if count > 0 else 0,
    }
    out.update(summary)
    return out


# --- describe_numeric_pandas.py -------------------------------------------------


def describe_numeric(config: Settings, series: pd.Series, summary: dict) -> None:
    series = series.dropna()
    vc = summary["value_counts_without_nan"]
    negative_index = vc.index < 0
    summary["n_negative"] = vc.loc[negative_index].sum()
    summary["p_negative"] = summary["n_negative"] / summary["n"]
    infinity_index = vc.index.isin([np.inf, -np.inf])
    summary["n_infinite"] = vc.loc[infinity_index].sum()
    summary["n_zeros"] = vc.loc[0] if 0 in vc.index else 0

    if isinstance(series.dtype, pd.api.extensions.ExtensionDtype) and pd.api.types.is_integer_dtype(series.dtype):
        summary.update(
            {
                "mean": series.mean(),
                "std": series.std(),
                "variance": series.var(),
                "min": series.min(),
                "max": series.max(),
                "kurtosis": series.kurt(),
                "skewness": series.skew(),
                "sum": series.sum(),
            }
        )
        present = series.astype(str(series.dtype).lower())
        finite = present
    else:
        present = series.values
        finite = present[np.isfinite(present)]
        index_values = vc.index.values
        if len(index_values):
            summary.update(
                {
                    "mean": np.average(index_values, weights=vc.values),
                    "std": np.std(present, ddof=1),
                    "variance": np.var(present, ddof=1),
                    "min": np.min(index_values),
                    "max": np.max(index_values),
                    "kurtosis": series.kurt(),
                    "skewness": series.skew(),
                    "sum": np.dot(index_values, vc.values),
                }
            )
        else:
            summary.update(
                {"mean": np.nan, "std": 0.0, "variance": 0.0, "min": np.nan, "max": np.nan,
                 "kurtosis": 0.0, "skewness": 0.0, "sum": 0}
            )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        summary["mad"] = np.median(np.abs(present - np.median(present)))
    if config.vars.num.chi_squared_threshold > 0.0:
        summary["chi_squared"] = chi_square(np.asarray(finite))  # keep the dtype: numpy bins integers differently
    summary["range"] = summary["max"] - summary["min"]
    summary.update(
        {f"{q:.0%}": v for q, v in series.quantile(config.vars.num.quantiles).to_dict().items()}
    )
    summary["iqr"] = summary["75%"] - summary["25%"]
    summary["cv"] = summary["std"] / summary["mean"] if summary["mean"] else np.nan
    summary["p_zeros"] = summary["n_zeros"] / summary["n"]
    summary["p_infinite"] = summary["n_infinite"] / summary["n"]
    summary["monotonic_increase"] = series.is_monotonic_increasing
    summary["monotonic_decrease"] = series.is_monotonic_decreasing
    summary["monotonic_increase_strict"] = summary["monotonic_increase"] and series.is_unique
    summary["monotonic_decrease_strict"] = summary["monotonic_decrease"] and series.is_unique
    if summary["monotonic_increase_strict"]:
        summary["monotonic"] = 2
    elif summary["monotonic_decrease_strict"]:
        summary["monotonic"] = -2
    elif summary["monotonic_increase"]:
        summary["monotonic"] = 1
    elif summary["monotonic_decrease"]:
        summary["monotonic"] = -1
    else:
        summary["monotonic"] = 0
    finite_vc = vc[~infinity_index]
    if len(finite_vc.index.values) > 0:
        summary.update(
            histogram_compute(config, finite_vc.index.values, summary["n_distinct"], weights=finite_vc.values)
        )


# --- describe_categorical_pandas.py ------------------------------------------


def _counter_to_series(counter: Counter) -> pd.Series:
    if not counter:
        return pd.Series([], dtype=object)
    items, counts = zip(*counter.most_common())
    return pd.Series(counts, index=items)


def _character_counts(vc: pd.Series) -> pd.Series:
    series = pd.Series(vc.index, index=vc, dtype=object)
    characters = series[series != ""].apply(list).explode()
    counts = pd.Series(characters.index, index=characters).dropna()
    if len(counts) > 0:
        counts = counts.groupby(level=0, sort=False).sum().sort_values(ascending=False)
        counts = counts[counts.index.str.len() > 0]
    return counts


def unicode_summary(vc: pd.Series) -> dict:
    # ydata-profiling only names Unicode blocks, scripts and categories when its
    # optional ``unicode`` extra is installed; by default every name is "(unknown)".
    character_counts = _character_counts(vc)
    summary: Dict[str, Any] = {
        "n_characters_distinct": len(character_counts),
        "n_characters": np.sum(character_counts.values),
        "character_counts": character_counts,
        "category_alias_values": {key: UNKNOWN for key in character_counts.keys()},
        "block_alias_values": {key: UNKNOWN for key in character_counts.keys()},
    }
    for prefix, count_key, n_key in (
        ("block_alias", "block_alias_counts", "n_block_alias"),
        ("script", "script_counts", "n_scripts"),
        ("category_alias", "category_alias_counts", "n_category"),
    ):
        totals: Counter = Counter()
        per_name: Dict[str, Counter] = {UNKNOWN: Counter()} if len(character_counts) else {}
        for char, n_char in character_counts.items():
            totals[UNKNOWN] += n_char
            per_name[UNKNOWN][char] += n_char
        summary[count_key] = _counter_to_series(totals)
        summary[n_key] = len(summary[count_key])
        summary[f"{prefix}_char_counts"] = {k: _counter_to_series(v) for k, v in per_name.items()}
    return summary


def word_summary(vc: pd.Series, stop_words: List[str]) -> dict:
    series = pd.Series(vc.index, index=vc, dtype=object)
    words = series.str.lower().str.split().explode().str.strip(string.punctuation + string.whitespace)
    word_counts = pd.Series(words.index, index=words)
    word_counts = word_counts[word_counts.index.notnull()]
    word_counts = word_counts.groupby(level=0, sort=False).sum().sort_values(ascending=False)
    if stop_words:
        word_counts = word_counts.loc[~word_counts.index.isin([w.lower() for w in stop_words])]
    return {"word_counts": word_counts} if not word_counts.empty else {}


def length_summary(vc: pd.Series) -> dict:
    series = pd.Series(vc.index, index=vc, dtype=object)
    length = series.str.len()
    length_counts = pd.Series(length.index, index=length)
    length_counts = length_counts.groupby(level=0, sort=False).sum().sort_values(ascending=False)
    return {
        "max_length": np.max(length_counts.index),
        "mean_length": np.average(length_counts.index, weights=length_counts.values)
        if not length_counts.empty
        else np.nan,
        "median_length": weighted_median(length_counts.index.values, weights=length_counts.values)
        if not length_counts.empty
        else np.nan,
        "min_length": np.min(length_counts.index),
        "length_histogram": length_counts,
    }


def _describe_strings(series: pd.Series, summary: dict) -> Tuple[pd.Series, pd.Series]:
    series = series.dropna().astype(str)
    vc = summary["value_counts_without_nan"]
    vc.index = vc.index.astype(str)
    return series, vc


def _string_extras(config: Settings, opts: Settings, vc: pd.Series, summary: dict) -> None:
    if opts.length:
        summary.update(length_summary(vc))
        summary.update(
            histogram_compute(
                config,
                summary["length_histogram"].index.values,
                len(summary["length_histogram"]),
                name="histogram_length",
                weights=summary["length_histogram"].values,
            )
        )
    if opts.characters:
        summary.update(unicode_summary(vc))
    if opts.words:
        summary.update(word_summary(vc, config.vars.cat.stop_words))


def describe_categorical(config: Settings, series: pd.Series, summary: dict) -> None:
    series, vc = _describe_strings(series, summary)
    summary["imbalance"] = imbalance_score(vc, len(vc))
    summary["first_rows"] = series.head(5)
    if config.vars.num.chi_squared_threshold > 0.0:
        summary["chi_squared"] = chi_square(histogram=vc.values)
    _string_extras(config, config.vars.cat, vc, summary)


def describe_text(config: Settings, series: pd.Series, summary: dict) -> None:
    series, vc = _describe_strings(series, summary)
    summary["first_rows"] = series.head(5)
    _string_extras(config, config.vars.text, vc, summary)


def describe_boolean(config: Settings, series: pd.Series, summary: dict) -> None:
    vc = summary["value_counts_without_nan"]
    if not vc.empty:
        summary.update({"top": vc.index[0], "freq": vc.iloc[0]})
        summary["imbalance"] = imbalance_score(vc, len(vc))
    else:
        summary.update({"top": np.nan, "freq": 0, "imbalance": 0})


def describe_date(config: Settings, series: pd.Series, summary: dict) -> np.ndarray:
    original = series.dropna()
    series = pd.to_datetime(original, format="mixed", errors="coerce")
    invalid = original[series.isna()]
    series = series.dropna()
    if summary["value_counts_without_nan"].empty:
        values = series.values
        summary.update({"min": pd.NaT, "max": pd.NaT, "range": 0})
    else:
        summary["min"] = pd.Timestamp.to_pydatetime(series.min())
        summary["max"] = pd.Timestamp.to_pydatetime(series.max())
        summary["range"] = summary["max"] - summary["min"]
        values = series.values.astype("datetime64[ns]").astype(np.int64) // 10**9
    if config.vars.num.chi_squared_threshold > 0.0:
        summary["chi_squared"] = chi_square(values)
    summary.update(histogram_compute(config, values, series.nunique()))
    summary.update(
        {
            "invalid_dates": invalid.nunique(),
            "n_invalid_dates": len(invalid),
            "p_invalid_dates": len(invalid) / summary["n"],
        }
    )
    return values


TYPE_DESCRIBERS = {
    "Numeric": describe_numeric,
    "Categorical": describe_categorical,
    "Text": describe_text,
    "Boolean": describe_boolean,
    "DateTime": describe_date,
}


def describe_series(config: Settings, series: pd.Series) -> dict:
    """summary_pandas.pandas_describe_1d: infer the type, then describe."""
    with pd.option_context("future.no_silent_downcasting", True):
        series = series.fillna(np.nan).infer_objects(copy=False)
    vtype, series = infer(series, config)
    summary: dict = {"type": vtype}
    describe_counts(series, summary)
    describe_generic(config, series, summary)
    summary = describe_supported(summary)
    if vtype in TYPE_DESCRIBERS and summary["hashable"]:
        # Infinite values make numpy warn (std, mad, ...); the NaN results are reported as-is.
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            TYPE_DESCRIBERS[vtype](config, series, summary)
    summary["cast_type"] = None
    return summary


# --- table_pandas.py ------------------------------------------------------------


def table_stats(config: Settings, df: pd.DataFrame, variables: dict) -> dict:
    n = len(df) if not df.empty else 0
    memory_size = df.memory_usage(deep=config.memory_deep).sum()
    table = {
        "n": n,
        "n_var": len(df.columns),
        "memory_size": memory_size,
        "record_size": float(memory_size) / n if n > 0 else 0,
        "n_cells_missing": 0,
        "n_vars_with_missing": 0,
        "n_vars_all_missing": 0,
    }
    for summary in variables.values():
        if "n_missing" in summary and summary["n_missing"] > 0:
            table["n_vars_with_missing"] += 1
            table["n_cells_missing"] += summary["n_missing"]
            if summary["n_missing"] == n:
                table["n_vars_all_missing"] += 1
    table["p_cells_missing"] = (
        table["n_cells_missing"] / (table["n"] * table["n_var"]) if table["n"] > 0 and table["n_var"] > 0 else 0
    )
    table["types"] = dict(Counter(v["type"] for v in variables.values()))
    return table


# --- correlations_pandas.py (auto) ------------------------------------------------


def _cramers_v(confusion: pd.DataFrame) -> float:
    if confusion.empty:
        return 0
    chi2 = stats.chi2_contingency_statistic(confusion.to_numpy(), correction=True)
    n = confusion.sum().sum()
    phi2 = chi2 / n
    r = confusion.shape[0]
    k = confusion.shape[1] if len(confusion.shape) > 1 else 1
    with np.errstate(divide="ignore", invalid="ignore"):
        phi2corr = max(0.0, phi2 - ((k - 1.0) * (r - 1.0)) / (n - 1.0))
        rcorr = r - ((r - 1.0) ** 2.0) / (n - 1.0)
        kcorr = k - ((k - 1.0) ** 2.0) / (n - 1.0)
        rkcorr = min((kcorr - 1.0), (rcorr - 1.0))
        return 1.0 if rkcorr == 0.0 else float(np.sqrt(phi2corr / rkcorr))


def auto_correlation(config: Settings, df: pd.DataFrame, variables: dict) -> Optional[pd.DataFrame]:
    threshold = config.categorical_maximum_correlation_distinct
    numerical = [k for k, v in variables.items() if v["type"] in {"Numeric", "TimeSeries"} and v["n_distinct"] > 1]
    categorical = [
        k for k, v in variables.items() if v["type"] in {"Categorical", "Boolean"} and 1 < v["n_distinct"] <= threshold
    ]
    if len(numerical + categorical) <= 1:
        return None
    n_bins = config.correlations["auto"].get("n_bins", 10)
    discretized = df.copy()
    for column in df.select_dtypes(include=np.number).columns:
        discretized.loc[:, column] = pd.cut(df[column], bins=n_bins, labels=False, duplicates="drop")
    columns = sorted(numerical + categorical)
    matrix = pd.DataFrame(np.ones((len(columns), len(columns))), index=columns, columns=columns)
    for a, b in itertools.combinations(columns, 2):
        if a not in categorical and b not in categorical:
            score = stats.spearman(df[a], df[b])
        else:
            x = discretized[a] if a in numerical else df[a]
            y = discretized[b] if b in numerical else df[b]
            score = _cramers_v(pd.crosstab(x, y))
        matrix.loc[a, b] = matrix.loc[b, a] = score
    return matrix


def get_correlations(config: Settings, df: pd.DataFrame, variables: dict) -> dict:
    out = {}
    auto = config.correlations.get("auto")
    if auto and auto.get("calculate", True):
        try:
            matrix = auto_correlation(config, df, variables)
        except (ValueError, AssertionError, TypeError, IndexError) as exc:
            warnings.warn(f"There was an attempt to calculate the auto correlation, but this failed: {exc}", stacklevel=2)
            matrix = None
        if matrix is not None and len(matrix) > 0:
            out["auto"] = matrix
    return out


def check_correlation(matrix: pd.DataFrame, threshold: float) -> Dict[str, List[str]]:
    cols = matrix.columns
    flags = abs(matrix.values) >= threshold
    np.fill_diagonal(flags, False)
    return {col: cols[flags[i]].values.tolist() for i, col in enumerate(cols) if any(flags[i])}


# --- duplicates / sample ------------------------------------------------------------


def get_duplicates(config: Settings, df: pd.DataFrame, supported: List[str]) -> Tuple[dict, Optional[pd.DataFrame]]:
    n_head = config.duplicates.head
    if n_head <= 0:
        return {}, None
    if not supported or len(df) == 0:
        return {"n_duplicates": 0, "p_duplicates": 0.0}, None
    key = config.duplicates.key
    if key in df.columns:
        raise ValueError(f"Duplicates key ({key}) may not be part of the DataFrame.")
    mask = df.duplicated(subset=supported, keep=False)
    duplicated = (
        df[mask]
        .rename_axis(index=lambda _: None)
        .groupby(supported, dropna=False, observed=True)
        .size()
        .reset_index(name=key)
    )
    metrics = {"n_duplicates": len(duplicated[key]), "p_duplicates": len(duplicated[key]) / len(df)}
    return metrics, duplicated.nlargest(n_head, key)


def get_samples(config: Settings, df: pd.DataFrame) -> List[dict]:
    if len(df) == 0:
        return []
    samples = []
    if config.samples.head > 0:
        samples.append({"id": "head", "data": df.head(config.samples.head), "name": "First rows", "caption": None})
    if config.samples.tail > 0:
        samples.append({"id": "tail", "data": df.tail(config.samples.tail), "name": "Last rows", "caption": None})
    return samples


# --- missing.py ------------------------------------------------------------------------

MISSING_DIAGRAMS = {
    "bar": ("Count", "A simple visualization of nullity by column.", 0),
    "matrix": (
        "Matrix",
        "Nullity matrix is a data-dense display which lets you quickly visually pick out patterns in data completion.",
        0,
    ),
    "heatmap": (
        "Heatmap",
        "The correlation heatmap measures nullity correlation: how strongly the presence or absence of one variable "
        "affects the presence of another.",
        2,
    ),
}


def active_missing_diagrams(config: Settings, table: dict) -> List[str]:
    active = []
    for name, (_, _, min_missing) in MISSING_DIAGRAMS.items():
        if not config.missing_diagrams.get(name, False) or table["n_vars_with_missing"] < min_missing:
            continue
        if name == "heatmap" and table["n_vars_with_missing"] - table["n_vars_all_missing"] < min_missing:
            continue
        active.append(name)
    return active


# --- describe.py -------------------------------------------------------------------------


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """dataframe_pandas.pandas_preprocess and utils.dataframe.rename_index."""
    df = df.copy()
    if "index" in df.columns:
        df = df.rename(columns={"index": "df_index"})
    df.columns = df.columns.map(str)
    return df


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def describe(config: Settings, df: pd.DataFrame) -> Dict[str, Any]:
    """Everything the report needs, except the charts (see ``charts.py``)."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"`df` must be a pandas.DataFrame, got {type(df)}")
    df = preprocess(df)
    date_start = utcnow()

    variables = {name: describe_series(config, series) for name, series in df.items()}
    if config.sort is not None:
        if str(config.sort).lower().startswith(("asc", "desc")):
            descending = str(config.sort).lower().startswith("desc")
            variables = dict(sorted(variables.items(), key=lambda x: x[0].casefold(), reverse=descending))
        else:
            raise ValueError('"sort" should be "ascending", "descending" or None.')

    supported = [c for c, d in variables.items() if d["type"] != "Unsupported"]
    interval = [c for c, d in variables.items() if d["type"] in {"Numeric", "TimeSeries"}]

    table = table_stats(config, df, variables)
    correlations = get_correlations(config, df, variables) if table["n"] != 0 else {}

    scatter_tasks: List[Tuple[str, str]] = []
    if config.interactions.continuous:
        targets = config.interactions.targets or interval
        scatter_tasks = [(x, y) for y in interval for x in targets]

    missing = active_missing_diagrams(config, table)
    samples = get_samples(config, df)
    metrics, duplicates = get_duplicates(config, df, supported)
    table.update(metrics)
    alerts = get_alerts(config, table, variables, correlations)

    return {
        "df": df,
        "date_start": date_start,
        "date_end": utcnow(),
        "table": table,
        "variables": variables,
        "interval_columns": interval,
        "scatter_tasks": scatter_tasks,
        "correlations": correlations,
        "missing": missing,
        "alerts": alerts,
        "sample": samples,
        "duplicates": duplicates,
    }
