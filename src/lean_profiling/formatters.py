"""Value formatting for the report (port of ydata-profiling's ``report/formatters.py``)."""

import decimal
import math
import re
from datetime import timedelta
from typing import Any, Optional, Union

from lean_profiling.alerts import Alert, fmt_percent  # noqa: F401  (re-exported)
from lean_profiling.html import escape


def fmt_bytesize(num: float, suffix: str = "B") -> str:
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f} Yi{suffix}"


def fmt_numeric(value: float, precision: int = 10) -> str:
    if value is None:
        return "N/A"
    formatted = f"{{:.{precision}g}}".format(value)
    for marker in ["e+", "e-"]:
        if marker in formatted:
            sign = "-" if marker in "e-" else ""
            formatted = formatted.replace(marker, " × 10<sup>") + "</sup>"
            formatted = formatted.replace("<sup>0", "<sup>")
            formatted = formatted.replace("<sup>", f"<sup>{sign}")
    return formatted


def fmt_number(value: int) -> str:
    return f"{value:n}"


def fmt(value: Any) -> str:
    if type(value) in [float, int]:
        return fmt_numeric(value)
    return escape(value)


def fmt_monotonic(value: int) -> str:
    names = {2: "Strictly increasing", 1: "Increasing", 0: "Not monotonic", -1: "Decreasing", -2: "Strictly decreasing"}
    if value not in names:
        raise ValueError("Value should be integer ranging from -2 to 2.")
    return names[value]


def fmt_timespan(num_seconds: Union[timedelta, int, float], detailed: bool = False, max_units: int = 3) -> str:
    """From humanfriendly, as vendored by ydata-profiling."""
    units = [
        (1e-9, "nanosecond"), (1e-6, "microsecond"), (1e-3, "millisecond"), (1, "second"), (60, "minute"),
        (3600, "hour"), (86400, "day"), (604800, "week"), (604800 * 52, "year"),
    ]

    def round_number(count: Any) -> str:
        text = f"{float(count):.2f}"
        return re.sub(r"\.$", "", re.sub("0+$", "", text))

    def pluralize(count: Any, singular: str) -> str:
        return f"{count} {singular if math.floor(float(count)) == 1 else singular + 's'}"

    seconds = num_seconds.total_seconds() if isinstance(num_seconds, timedelta) else float(num_seconds)
    if seconds < 60 and not detailed:
        return pluralize(round_number(seconds), "second")
    result = []
    remaining = decimal.Decimal(str(seconds))
    relevant = list(reversed(units[0 if detailed else 3 :]))
    for divider_value, name in relevant:
        divider = decimal.Decimal(str(divider_value))
        count: Any = remaining / divider
        remaining %= divider
        count = int(count) if (divider_value, name) != relevant[-1] else round_number(count)
        if count not in (0, "0"):
            result.append(pluralize(count, name))
    if len(result) == 1:
        return result[0]
    if not detailed:
        result = result[:max_units]
    return ", ".join(result[:-1]) + " and " + result[-1] if len(result) > 1 else (result[0] if result else "")


def help_badge(title: str, url: Optional[str] = None) -> str:
    badge = (
        f'<span class="badge text-bg-secondary float-end" data-bs-toggle="tooltip" data-bs-placement="right" '
        f'data-bs-title="{title}">?</span>'
    )
    return f'<a href="{url}">{badge}</a>' if url else badge


def alert_badge(alert: Alert) -> str:
    """Alert.fmt: the badge shown beside a variable."""
    hint = ""
    if alert.name == "HIGH_CORRELATION":
        fields = alert.values["fields"]
        hint = (
            f'data-bs-toggle="tooltip" data-bs-placement="right" data-bs-title="This variable has a high '
            f'{alert.values["corr"]} correlation with {len(fields)} fields: {", ".join(fields)}"'
        )
    return f'<span class="badge text-bg-{alert.style}" {hint}>{alert.type_name}</span>'


def alert_html(alert: Alert) -> str:
    """templates/alerts/alert_*.html: the line in the overview's alerts table."""
    v = alert.values
    link = f'<a href="#pp_var_{alert.anchor_id}"><code>{escape(alert.column_name)}</code></a>'
    name = alert.name
    if name == "CONSTANT":
        return f'{link} has constant value "{escape(v["value_counts_without_nan"].index[0])}"'
    if name == "CONSTANT_LENGTH":
        return f"{link} has constant length"
    if name == "DUPLICATES":
        return f'Dataset has {v["n_duplicates"]} ({fmt_percent(v["p_duplicates"])}) <a href="#duplicate">duplicate rows</a>'
    if name == "EMPTY":
        return "Dataset is empty"
    if name == "HIGH_CARDINALITY":
        return f'{link} has a high cardinality: {v["n_distinct"]} distinct values'
    if name == "HIGH_CORRELATION":
        fields = v["fields"]
        out = f'{link} is highly {v["corr"]} correlated with <code>{escape(fields[0])}</code>'
        if len(fields) > 1:
            out += (
                f' and <span data-bs-toggle="tooltip" data-bs-placement="right" '
                f'data-bs-title="{escape(", ".join(fields))}">{len(fields) - 1} other fields</span>'
            )
        return out
    if name == "IMBALANCE":
        return f'{link} is highly imbalanced ({fmt_percent(v["imbalance"])})'
    if name == "INFINITE":
        return f'{link} has {v["n_infinite"]} ({fmt_percent(v["p_infinite"])}) infinite values'
    if name == "MISSING":
        return f'{link} has {v["n_missing"]} ({fmt_percent(v["p_missing"])}) missing values'
    if name == "SKEWED":
        return f'{link} is highly skewed (γ1 = {fmt_numeric(v["skewness"])})'
    if name == "TYPE_DATE":
        return f"{link} only contains datetime values, but is categorical. Consider applying <code>pd.to_datetime()</code>"
    if name == "UNIFORM":
        return f"{link} is uniformly distributed"
    if name == "UNIQUE":
        return f"{link} has unique values"
    if name == "UNSUPPORTED":
        return f"{link} is an unsupported type, check if it needs cleaning or further analysis"
    if name == "ZEROS":
        return f'{link} has {v["n_zeros"]} ({fmt_percent(v["p_zeros"])}) zeros'
    return ""
