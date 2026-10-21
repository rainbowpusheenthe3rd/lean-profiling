"""``ProfileReport``: the report structure, charts, HTML page and JSON export.

Mirrors ydata-profiling's ``report/structure`` package and ``ProfileReport`` API
(``to_file``, ``to_html``, ``to_json``, ``get_description``).
"""

import re
import unicodedata
from importlib import resources
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import quote

import numpy as np
import pandas as pd

from lean_profiling import charts
from lean_profiling.alerts import var_id
from lean_profiling.config import Settings
from lean_profiling.describe import MISSING_DIAGRAMS, describe
from lean_profiling.formatters import (
    alert_badge,
    fmt,
    fmt_bytesize,
    fmt_monotonic,
    fmt_number,
    fmt_numeric,
    fmt_percent,
    fmt_timespan,
    help_badge,
)
from lean_profiling.html import (
    HTML,
    Alerts,
    Container,
    DataFrameTable,
    Dropdown,
    FrequencyTable,
    FrequencyTableSmall,
    Image,
    Renderable,
    Table,
    Variable,
    VariableInfo,
    escape,
    image_or_empty,
)
from lean_profiling.serialize import to_json

__version__ = "1.0.0"
REPO_URL = "https://github.com/rainbowpusheenthe3rd/lean-profiling"


def slugify(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")


# --- frequency tables (presentation/frequency_table_utils.py) ------------------------------


def freq_table(freqtable: pd.Series, n: int, max_number_to_print: int) -> List[dict]:
    if max_number_to_print > n:
        max_number_to_print = n
    if max_number_to_print < len(freqtable):
        freq_other = np.sum(freqtable.iloc[max_number_to_print:])
        min_freq = freqtable.values[max_number_to_print]
    else:
        freq_other = 0
        min_freq = 0
    freq_missing = n - np.sum(freqtable)
    if len(freqtable) == 0:
        return []
    max_freq = max(freqtable.values[0], freq_other, freq_missing)
    if max_freq == 0:
        return []
    rows = [
        {"label": label, "width": freq / max_freq, "count": freq, "percentage": float(freq) / n, "extra_class": ""}
        for label, freq in freqtable.iloc[0:max_number_to_print].items()
    ]
    if freq_other > min_freq:
        rows.append(
            {
                "label": f"Other values ({freqtable.count() - max_number_to_print})",
                "width": freq_other / max_freq,
                "count": freq_other,
                "percentage": min(float(freq_other) / n, 1.0),
                "extra_class": "other",
            }
        )
    if freq_missing > min_freq:
        rows.append(
            {
                "label": "(Missing)",
                "width": freq_missing / max_freq,
                "count": freq_missing,
                "percentage": float(freq_missing) / n,
                "extra_class": "missing",
            }
        )
    return rows


def extreme_obs_table(freqtable: pd.Series, number_to_print: int, n: int) -> List[dict]:
    obs = freqtable.iloc[:number_to_print]
    max_freq = obs.max()
    return [
        {"label": label, "width": freq / max_freq if max_freq != 0 else 0, "count": freq,
         "percentage": float(freq) / n, "extra_class": ""}
        for label, freq in obs.items()
    ]


# --- per-variable rendering (report/structure/variables) ----------------------------------------


class VariableRenderer:
    def __init__(self, config: Settings, name: str, summary: dict, alerts: list):
        self.config = config
        self.name = name
        self.s = summary
        self.varid = var_id(name)
        own = [a for a in alerts if a.column_name == name]
        self.alerts = [alert_badge(a) for a in own]
        self.alert_fields = {f for a in own for f in a.fields}
        self.ignore = config.reject_variables and any(a.name == "REJECTED" for a in own)
        descriptions = config.variables.descriptions
        self.description = descriptions.get(name, "") if config.show_variable_description else ""
        self.precision = config.report.precision

    def info(self, var_type: str) -> VariableInfo:
        return VariableInfo(self.varid, self.name, var_type, self.alerts, self.description)

    def row(self, name: str, value: str, field: Optional[str] = None, **extra: Any) -> dict:
        return {"name": name, "value": value, "alert": field in self.alert_fields if field else False, **extra}

    def num(self, key: str) -> str:
        return fmt_numeric(self.s[key], precision=self.precision)

    def common(self) -> Dict[str, List[dict]]:
        s, c = self.s, self.config
        out = {"freq": freq_table(s["value_counts_without_nan"], s["n"], c.n_freq_table_max)}
        if "value_counts_index_sorted" in s:
            out["first"] = extreme_obs_table(s["value_counts_index_sorted"], c.n_extreme_obs, s["n"])
            out["last"] = extreme_obs_table(s["value_counts_index_sorted"][::-1], c.n_extreme_obs, s["n"])
        return out

    def distinct_missing_rows(self) -> List[dict]:
        s = self.s
        return [
            self.row("Distinct", fmt(s["n_distinct"]), "n_distinct"),
            self.row("Distinct (%)", fmt_percent(s["p_distinct"]), "p_distinct"),
            self.row("Missing", fmt(s["n_missing"]), "n_missing"),
            self.row("Missing (%)", fmt_percent(s["p_missing"]), "p_missing"),
        ]

    def render(self) -> Variable:
        method = {
            "Numeric": self.numeric,
            "Categorical": self.categorical,
            "Boolean": self.boolean,
            "DateTime": self.date,
            "Text": self.text,
        }.get(self.s["type"], self.generic)
        top, bottom = method()
        return Variable(top, bottom, anchor_id=self.varid, name=self.name, ignore=self.ignore)

    # render_real.py
    def numeric(self):
        s, c, vid = self.s, self.config, self.varid
        tables = self.common()
        table1 = Table(
            self.distinct_missing_rows()
            + [
                self.row("Infinite", fmt(s["n_infinite"]), "n_infinite"),
                self.row("Infinite (%)", fmt_percent(s["p_infinite"]), "p_infinite"),
                self.row("Mean", self.num("mean")),
            ]
        )
        table2 = Table(
            [
                self.row("Minimum", self.num("min")),
                self.row("Maximum", self.num("max")),
                self.row("Zeros", fmt(s["n_zeros"]), "n_zeros"),
                self.row("Zeros (%)", fmt_percent(s["p_zeros"]), "p_zeros"),
                self.row("Negative", fmt(s["n_negative"])),
                self.row("Negative (%)", fmt_percent(s["p_negative"])),
                self.row("Memory size", fmt_bytesize(s["memory_size"])),
            ]
        )
        hist = s.get("histogram", [])
        mini = charts.histogram(c, *hist, mini=True) if len(hist) else None
        top = Container(
            [self.info("Real number (&Ropf;)"), table1, table2,
             image_or_empty(mini, alt="Mini histogram", name="Mini Histogram", anchor_id=f"{vid}minihistogram")],
            "grid",
        )
        quantiles = Table(
            [
                {"name": "Minimum", "value": self.num("min")},
                {"name": "5-th percentile", "value": self.num("5%")},
                {"name": "Q1", "value": self.num("25%")},
                {"name": "median", "value": self.num("50%")},
                {"name": "Q3", "value": self.num("75%")},
                {"name": "95-th percentile", "value": self.num("95%")},
                {"name": "Maximum", "value": self.num("max")},
                {"name": "Range", "value": self.num("range")},
                {"name": "Interquartile range (IQR)", "value": self.num("iqr")},
            ],
            name="Quantile statistics",
        )
        descriptive = Table(
            [
                {"name": "Standard deviation", "value": self.num("std")},
                {"name": "Coefficient of variation (CV)", "value": self.num("cv")},
                {"name": "Kurtosis", "value": self.num("kurtosis")},
                {"name": "Mean", "value": self.num("mean")},
                {"name": "Median Absolute Deviation (MAD)", "value": self.num("mad")},
                {"name": "Skewness", "value": self.num("skewness")},
                {"name": "Sum", "value": self.num("sum")},
                {"name": "Variance", "value": self.num("variance")},
                {"name": "Monotonicity", "value": fmt_monotonic(s["monotonic"])},
            ],
            name="Descriptive statistics",
        )
        statistics = Container([quantiles, descriptive], "grid", anchor_id=f"{vid}statistics", name="Statistics")
        big = charts.histogram(c, *hist) if len(hist) else None
        caption = f"<strong>Histogram with fixed size bins</strong> (bins={len(hist[1]) - 1})" if len(hist) else None
        histogram = image_or_empty(big, alt="Histogram", caption=caption, name="Histogram", anchor_id=f"{vid}histogram")
        common_values = FrequencyTable(tables["freq"], name="Common values", anchor_id=f"{vid}common_values")
        extremes = Container(
            [
                FrequencyTable(tables["first"], name=f"Minimum {c.n_extreme_obs} values", anchor_id=f"{vid}firstn"),
                FrequencyTable(tables["last"], name=f"Maximum {c.n_extreme_obs} values", anchor_id=f"{vid}lastn"),
            ],
            "tabs",
            name="Extreme values",
            anchor_id=f"{vid}extreme_values",
        )
        bottom = Container([statistics, histogram, common_values, extremes], "tabs", anchor_id=f"{vid}bottom")
        return top, bottom

    # render_categorical.py helpers
    def unique_table(self) -> Table:
        s = self.s
        return Table(
            [
                self.row(
                    "Unique",
                    fmt_number(s["n_unique"]),
                    "n_unique",
                    hint=help_badge(
                        "The number of unique values (all values that occur exactly once in the dataset)."
                    ),
                ),
                self.row("Unique (%)", fmt_percent(s["p_unique"]), "p_unique"),
            ],
            name="Unique",
            anchor_id=f"{self.varid}_unique_stats",
        )

    def length_items(self):
        s, vid = self.s, self.varid
        table = Table(
            [
                self.row("Max length", fmt_number(s["max_length"])),
                self.row("Median length", fmt_number(s["median_length"])),
                self.row("Mean length", self.num("mean_length")),
                self.row("Min length", fmt_number(s["min_length"])),
            ],
            name="Length",
            anchor_id=f"{vid}lengthstats",
        )
        hist = s["histogram_length"]
        image = charts.histogram(self.config, *hist) if len(hist) else None
        histo = image_or_empty(image, alt="length histogram", name="Length",
                               caption="Histogram of lengths of the category", anchor_id=f"{vid}length")
        return table, histo

    def unicode_items(self):
        s, vid, n_max = self.s, self.varid, self.config.n_freq_table_max

        def table(series: pd.Series, name: str, anchor: str) -> FrequencyTable:
            return FrequencyTable(freq_table(series, series.sum(), n_max), name=name, anchor_id=anchor)

        def per_name(counts: dict, anchor: str, sort: bool, rename: bool = False) -> List[FrequencyTable]:
            items = sorted(counts.items(), key=lambda x: -len(x[1])) if sort else list(counts.items())
            out = []
            for key, series in items:
                label = key.replace("_", " ") if rename else key
                out.append(table(series, label, f"{vid}{anchor}_{label}"))
            return out

        def group(items: List[FrequencyTable], name: str, anchor: str) -> Container:
            return Container(items, "batch_grid", name=name, anchor_id=anchor, batch_size=1, subtitles=True)

        categories = [
            table(s["category_alias_counts"], "Most occurring categories", f"{vid}category_long_values"),
            group(per_name(s["category_alias_char_counts"], "category_alias_values", True, rename=True),
                  "Most frequent character per category", f"{vid}categories"),
        ]
        scripts = [
            table(s["script_counts"], "Most occurring scripts", f"{vid}script_values"),
            group(per_name(s["script_char_counts"], "script_values", True), "Most frequent character per script",
                  f"{vid}scripts"),
        ]
        blocks = [
            table(s["block_alias_counts"], "Most occurring blocks", f"{vid}block_alias_values"),
            group(per_name(s["block_alias_char_counts"], "block_alias_values", False),
                  "Most frequent character per block", f"{vid}blocks"),
        ]
        overview = Table(
            [
                self.row("Total characters", fmt_number(s["n_characters"])),
                self.row("Distinct characters", fmt_number(s["n_characters_distinct"])),
                self.row("Distinct categories", fmt_number(s["n_category"]),
                         hint=help_badge("Unicode categories (click for more information)",
                                         "https://en.wikipedia.org/wiki/Unicode_character_property#General_Category")),
                self.row("Distinct scripts", fmt_number(s["n_scripts"]),
                         hint=help_badge("Unicode scripts (click for more information)",
                                         "https://en.wikipedia.org/wiki/Script_(Unicode)#List_of_scripts_in_Unicode")),
                self.row("Distinct blocks", fmt_number(s["n_block_alias"]),
                         hint=help_badge("Unicode blocks (click for more information)",
                                         "https://en.wikipedia.org/wiki/Unicode_block")),
            ],
            name="Characters and Unicode",
            caption="The Unicode Standard assigns character properties to each code point, which can be used to "
            "analyse textual variables. ",
        )
        characters = Container(
            [FrequencyTable(freq_table(s["character_counts"], s["n_characters"], n_max),
                            name="Most occurring characters", anchor_id=f"{vid}character_frequency")],
            "named_list", name="Characters", anchor_id=f"{vid}characters",
        )
        tabs = Container(
            [
                characters,
                Container(categories, "named_list", name="Categories", anchor_id=f"{vid}categories"),
                Container(scripts, "named_list", name="Scripts", anchor_id=f"{vid}scripts"),
                Container(blocks, "named_list", name="Blocks", anchor_id=f"{vid}blocks"),
            ],
            "tabs", name="Unicode", anchor_id=f"{vid}unicode",
        )
        return overview, tabs

    def sample_table(self) -> Table:
        names = ("1st row", "2nd row", "3rd row", "4th row", "5th row")
        return Table([self.row(n, fmt(v)) for n, v in zip(names, self.s["first_rows"])], name="Sample")

    def string_overview(self, opts) -> tuple:
        s = self.s
        items: List[Renderable] = []
        length_histo = unicode_tab = None
        if opts.length and "max_length" in s:
            length_table, length_histo = self.length_items()
            items.append(length_table)
        if opts.characters and "category_alias_counts" in s:
            unicode_table, unicode_tab = self.unicode_items()
            items.append(unicode_table)
        items.append(self.unique_table())
        items.append(self.sample_table())
        overview = Container(items, "batch_grid", name="Overview", anchor_id=f"{self.varid}overview",
                             batch_size=len(items), titles=False)
        return overview, length_histo, unicode_tab

    def words_table(self) -> FrequencyTable:
        wc = self.s["word_counts"]
        return FrequencyTable(freq_table(wc, wc.sum(), 10), name="Common words", anchor_id=f"{self.varid}cwo")

    # render_categorical.py
    def categorical(self):
        s, c, vid = self.s, self.config, self.varid
        tables = self.common()
        table = Table(self.distinct_missing_rows() + [self.row("Memory size", fmt_bytesize(s["memory_size"]))])
        mini = FrequencyTableSmall(freq_table(s["value_counts_without_nan"], s["count"], c.vars.cat.n_obs), varid=vid)
        top = Container([self.info(s["type"]), table, mini], "grid")
        overview, length_histo, unicode_tab = self.string_overview(c.vars.cat)
        string_items: List[Renderable] = [
            FrequencyTable(tables["freq"], name="Common Values", anchor_id=f"{vid}common_values")
        ]
        if length_histo is not None:
            string_items.append(length_histo)
        cat_freq = c.plot.cat_freq
        if cat_freq.show and cat_freq.max_unique > 0 and s["n_distinct"] <= cat_freq.max_unique:
            string_items.append(
                Image(charts.cat_frequency_plot(c, s["value_counts_without_nan"]), alt="Common Values (Plot)",
                      name="Common Values (Plot)", anchor_id=f"{vid}cat_frequency_plot")
            )
        bottom_items: List[Renderable] = [
            overview,
            Container(string_items, "batch_grid", name="Categories", anchor_id=f"{vid}string", batch_size=1),
        ]
        if c.vars.cat.words and "word_counts" in s:
            bottom_items.append(Container([self.words_table()], "grid", name="Words", anchor_id=f"{vid}word"))
        if unicode_tab is not None:
            bottom_items.append(Container([unicode_tab], "grid", name="Characters", anchor_id=f"{vid}characters"))
        return top, Container(bottom_items, "tabs", anchor_id=f"{vid}bottom")

    # render_text.py
    def text(self):
        s, c, vid = self.s, self.config, self.varid
        self.common()
        table = Table(self.distinct_missing_rows() + [self.row("Memory size", fmt_bytesize(s["memory_size"]))])
        top_items: List[Renderable] = [self.info(s["type"]), table]
        cloud = None
        if c.vars.text.words and "word_counts" in s:
            cloud = charts.word_cloud(s["word_counts"])
            top_items.append(Image(cloud, alt="Mini wordcloud"))
        overview, _, unicode_tab = self.string_overview(c.vars.text)
        bottom_items: List[Renderable] = [overview]
        if cloud is not None:
            bottom_items.append(
                Container([self.words_table(), Image(cloud, alt="Wordcloud")], "grid", name="Words",
                          anchor_id=f"{vid}word")
            )
        if unicode_tab is not None:
            bottom_items.append(Container([unicode_tab], "grid", name="Characters", anchor_id=f"{vid}characters"))
        return Container(top_items, "grid"), Container(bottom_items, "tabs", anchor_id=f"{vid}bottom")

    # render_boolean.py
    def boolean(self):
        s, c, vid = self.s, self.config, self.varid
        tables = self.common()
        table = Table(self.distinct_missing_rows() + [self.row("Memory size", fmt_bytesize(s["memory_size"]))])
        mini = FrequencyTableSmall(freq_table(s["value_counts_without_nan"], s["n"], c.vars.bool.n_obs), varid=vid)
        top = Container([self.info("Boolean"), table, mini], "grid")
        items: List[Renderable] = [
            FrequencyTable(tables["freq"], name="Common Values (Table)", anchor_id=f"{vid}frequency_table")
        ]
        if c.plot.cat_freq.show and c.plot.cat_freq.max_unique > 0:
            items.append(
                Image(charts.cat_frequency_plot(c, s["value_counts_without_nan"]), alt="Common Values (Plot)",
                      name="Common Values (Plot)", anchor_id=f"{vid}cat_frequency_plot")
            )
        return top, Container(items, "tabs", anchor_id=f"{vid}bottom")

    # render_date.py
    def date(self):
        s, c, vid = self.s, self.config, self.varid
        no_alert = {"alert": False}
        table1 = Table(
            [
                {"name": "Distinct", "value": fmt(s["n_distinct"]), **no_alert},
                {"name": "Distinct (%)", "value": fmt_percent(s["p_distinct"]), **no_alert},
                {"name": "Missing", "value": fmt(s["n_missing"]), **no_alert},
                {"name": "Missing (%)", "value": fmt_percent(s["p_missing"]), **no_alert},
                {"name": "Memory size", "value": fmt_bytesize(s["memory_size"]), **no_alert},
            ]
        )
        table2 = Table(
            [
                {"name": "Minimum", "value": fmt(s["min"]), **no_alert},
                {"name": "Maximum", "value": fmt(s["max"]), **no_alert},
                {"name": "Invalid dates", "value": fmt(s["n_invalid_dates"]), **no_alert},
                {"name": "Invalid dates (%)", "value": fmt_percent(s["p_invalid_dates"]), **no_alert},
            ]
        )
        hist = s.get("histogram", [])
        mini = charts.histogram(c, *hist, date=True, mini=True) if len(hist) else None
        top = Container(
            [self.info("Date"), table1, table2,
             image_or_empty(mini, alt="Mini histogram", name="Mini Histogram", anchor_id=f"{vid}minihistogram")],
            "grid",
        )
        big = charts.histogram(c, *hist, date=True) if len(hist) else None
        caption = f"<strong>Histogram with fixed size bins</strong> (bins={len(hist[1]) - 1 if len(hist) else 0})" \
            if len(hist) else None
        histogram = image_or_empty(big, alt="Histogram", caption=caption, name="Histogram", anchor_id=f"{vid}histogram")
        return top, Container([histogram], "tabs", anchor_id=vid)

    # render_generic.py
    def generic(self):
        s = self.s
        table = Table(
            [
                self.row("Missing", fmt(s["n_missing"]), "n_missing"),
                self.row("Missing (%)", fmt_percent(s["p_missing"]), "p_missing"),
                self.row("Memory size", fmt_bytesize(s["memory_size"])),
            ]
        )
        return Container([self.info(s["cast_type"] or "Unsupported"), table, HTML("")], "grid"), None


# --- report ------------------------------------------------------------------------------------


class ProfileReport:
    """Profile a pandas DataFrame into an HTML report and a JSON description.

    Example:
        >>> report = ProfileReport(pd.read_csv("training.csv"), title="Training data")
        >>> report.to_file("profiling.json")
        >>> report.to_file("report.html")
    """

    def __init__(self, df: pd.DataFrame, minimal: bool = False, title: Optional[str] = None,
                 progress_bar: bool = False, **kwargs: Any):
        # ``progress_bar`` is accepted for ydata-profiling compatibility; there is no progress bar.
        if title is not None:
            kwargs["title"] = title
        self.config = Settings.build(minimal=minimal, **kwargs)
        self.df = df
        self._description: Optional[dict] = None
        self._charts: Optional[dict] = None
        self._html: Optional[str] = None
        self._json: Optional[str] = None

    # Description and charts are computed once and shared by the HTML and JSON outputs.

    @property
    def description_set(self) -> dict:
        if self._description is None:
            self._description = describe(self.config, self.df)
        return self._description

    def _shared_charts(self) -> dict:
        if self._charts is None:
            d, c = self.description_set, self.config
            df = d["df"]
            scatter: Dict[str, Dict[str, str]] = {}
            # ydata-profiling builds {x: {last_y: None}} first, so each x's last y comes first.
            for x, y in d["scatter_tasks"]:
                scatter[x] = {y: None}
            for x, y in d["scatter_tasks"]:
                frame = df[[x]].dropna() if x == y else df[[x, y]].dropna()
                scatter[x][y] = charts.scatter_pairwise(c, frame[x], frame[y], x, y)
            missing = {}
            for key in d["missing"]:
                name, caption, _ = MISSING_DIAGRAMS[key]
                image = self._missing_chart(key, df)
                if image is not None:
                    missing[key] = {"name": name, "caption": caption, "matrix": image}
            self._charts = {"scatter": scatter, "missing": missing}
        return self._charts

    def _missing_chart(self, key: str, df: pd.DataFrame) -> Optional[str]:
        c = self.config
        columns = list(df.columns)
        if len(df) == 0 or not columns:
            # ydata-profiling's plots fail here and the diagram is skipped with a warning.
            return None
        if key == "bar":
            return charts.missing_bar(c, len(df) - df.isnull().sum(), len(df), columns)
        if key == "matrix":
            return charts.missing_matrix(c, df.notnull().values, columns, len(df))
        keep = [i for i, v in enumerate(np.var(df.isnull(), axis="rows")) if v > 0]
        sub = df.iloc[:, keep]
        corr = sub.isnull().corr()
        mask = np.zeros_like(corr, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        return charts.missing_heatmap(c, corr, mask, list(sub.columns))

    def get_description(self) -> dict:
        """The full description, in the layout of ``profiling.json``."""
        d, shared = self.description_set, self._shared_charts()
        return {
            "analysis": {"title": self.config.title, "date_start": d["date_start"], "date_end": d["date_end"]},
            "time_index_analysis": None,
            "table": d["table"],
            "variables": d["variables"],
            "scatter": shared["scatter"],
            "correlations": d["correlations"],
            "missing": shared["missing"],
            "alerts": [repr(a) for a in d["alerts"]],
            "package": {"lean_profiling_version": __version__, "lean_profiling_config": self.config.json()},
            "sample": d["sample"],
            "duplicates": d["duplicates"],
        }

    def get_duplicates(self) -> Optional[pd.DataFrame]:
        return self.description_set["duplicates"]

    def get_rejected_variables(self) -> set:
        return {a.column_name for a in self.description_set["alerts"] if a.name == "REJECTED"}

    def to_json(self) -> str:
        if self._json is None:
            self._json = to_json(self.get_description())
        return self._json

    def to_html(self) -> str:
        if self._html is None:
            self._html = self._render_html()
        return self._html

    def to_file(self, output_file: Union[str, Path], silent: bool = True) -> None:
        output_file = Path(output_file)
        if output_file.suffix == ".json":
            data = self.to_json()
        else:
            if output_file.suffix != ".html":
                output_file = output_file.with_suffix(".html")
            data = self.to_html()
        output_file.write_text(data, encoding="utf-8")

    # --- HTML -----------------------------------------------------------------------------------

    def _overview(self) -> Container:
        d, c = self.description_set, self.config
        t = d["table"]
        metrics = [
            {"name": "Number of variables", "value": fmt_number(t["n_var"])},
            {"name": "Number of observations", "value": fmt_number(t["n"])},
            {"name": "Missing cells", "value": fmt_number(t["n_cells_missing"])},
            {"name": "Missing cells (%)", "value": fmt_percent(t["p_cells_missing"])},
        ]
        if "n_duplicates" in t:
            metrics += [
                {"name": "Duplicate rows", "value": fmt_number(t["n_duplicates"])},
                {"name": "Duplicate rows (%)", "value": fmt_percent(t["p_duplicates"])},
            ]
        metrics += [
            {"name": "Total size in memory", "value": fmt_bytesize(t["memory_size"])},
            {"name": "Average record size in memory", "value": fmt_bytesize(t["record_size"])},
        ]
        types = Table(
            [{"name": str(k), "value": fmt_numeric(v, precision=c.report.precision)} for k, v in t["types"].items()],
            name="Variable types",
        )
        items: List[Renderable] = [
            Container([Table(metrics, name="Dataset statistics"), types], "grid", anchor_id="dataset_overview",
                      name="Overview")
        ]
        meta = dict(c.dataset)
        if any(len(str(v)) > 0 for v in meta.values()):
            about = [{"name": k.capitalize(), "value": fmt(meta[k])} for k in ["description", "creator", "author"]
                     if meta.get(k)]
            about.append({"name": "URL", "value": f'<a href="{meta.get("url", "")}">{meta.get("url", "")}</a>'})
            if meta.get("copyright_holder"):
                about.append({"name": "Copyright",
                              "value": fmt(f"(c) {meta['copyright_holder']} {meta.get('copyright_year', '')}")})
            items.append(Container([Table(about, name="Dataset", anchor_id="metadata_dataset")], "grid",
                                   name="Dataset", anchor_id="dataset"))
        descriptions = c.variables.descriptions
        if descriptions:
            items.append(Container(
                [Table([{"name": k, "value": fmt(v)} for k, v in descriptions.items()], name="Variable descriptions",
                       anchor_id="variable_definition_table")],
                "grid", name="Variables", anchor_id="variable_descriptions"))
        alerts = d["alerts"]
        if alerts:
            count = len([a for a in alerts if a.name != "REJECTED"])
            items.append(Alerts(alerts, name=f"Alerts ({count})", anchor_id="alerts"))
        config_json = c.json()
        reproduction = Table(
            [
                {"name": "Analysis started", "value": fmt(d["date_start"])},
                {"name": "Analysis finished", "value": fmt(d["date_end"])},
                {"name": "Duration", "value": fmt_timespan(d["date_end"] - d["date_start"])},
                {"name": "Software version", "value": f'<a href="{REPO_URL}">lean-profiling v{__version__}</a>'},
                {"name": "Download configuration",
                 "value": f'<a download="config.json" href="data:text/plain;charset=utf-8,{quote(config_json)}">'
                          "config.json</a>"},
            ],
            name="Reproduction",
            anchor_id="overview_reproduction",
        )
        items.append(Container([reproduction], "grid", name="Reproduction", anchor_id="reproduction"))
        return Container(items, "overview_tabs", name="Overview", anchor_id="overview")

    def _sections(self) -> List[Renderable]:
        d, c = self.description_set, self.config
        shared = self._shared_charts()
        sections: List[Renderable] = [self._overview()]

        if d["variables"]:
            variables = [VariableRenderer(c, name, s, d["alerts"]).render() for name, s in d["variables"].items()]
            sections.append(Dropdown("Variables", "variables-dropdown", list(d["variables"]),
                                     Container(variables, "accordion", name="Variables", anchor_id="variables"),
                                     anchor_id="variables-dropdown"))

        scatter_items = []
        for x, ys in shared["scatter"].items():
            items = [
                Image(plot, alt=f"{x} x {y}", anchor_id=f"interactions_{slugify(x)}_{slugify(y)}", name=y)
                for y, plot in ys.items()
            ]
            scatter_items.append(Container(items, "tabs" if len(items) <= 10 else "select", name=x,
                                           nested=len(shared["scatter"]) > 10, anchor_id=f"interactions_{slugify(x)}"))
        if scatter_items:
            sections.append(Container(scatter_items, "tabs" if len(scatter_items) <= 10 else "select",
                                      name="Interactions", anchor_id="interactions"))

        corr_items = []
        for key, matrix in d["correlations"].items():
            diagram = Image(charts.correlation_matrix(c, matrix), alt="Auto",
                            anchor_id=f"{key}_diagram", name="Heatmap" if c.correlation_table else "Auto",
                            classes="correlation-diagram")
            if c.correlation_table:
                table = DataFrameTable(
                    matrix.to_html(classes="correlation-table table table-striped", float_format="{:.3f}".format),
                    "correlation-table-container", name="Table", anchor_id=f"{key}_table",
                )
                corr_items.append(Container([diagram, table], "tabs", name="Auto", anchor_id=f"{key}_diagram_table"))
            else:
                corr_items.append(diagram)
        if corr_items:
            sections.append(Container(corr_items, "tabs", name="Correlations", anchor_id="correlations_tab"))

        missing_items = [
            Image(item["matrix"], alt=item["name"], name=item["name"], anchor_id=key, caption=item["caption"])
            for key, item in shared["missing"].items()
        ]
        if missing_items:
            sections.append(Container(missing_items, "tabs", name="Missing values", anchor_id="missing"))

        samples = [
            DataFrameTable(s["data"].to_html(classes="sample table table-striped"), "sample-container",
                           caption=s["caption"], name=s["name"], anchor_id=s["id"])
            for s in d["sample"]
        ]
        if samples:
            sections.append(Container(samples, "tabs", name="Sample", anchor_id="sample"))

        dups = d["duplicates"]
        if dups is not None and len(dups) > 0:
            sections.append(Container(
                [DataFrameTable(dups.to_html(classes="duplicate table table-striped"), "duplicate-container",
                                name="Most frequently occurring", anchor_id="duplicates")],
                "batch_grid", batch_size=1, name="Duplicate rows", anchor_id="duplicate"))
        return sections

    def _render_html(self) -> str:
        c = self.config
        d = self.description_set
        sections = self._sections()
        body = Container(sections, "sections", name="Root", full_width=c.html.full_width).render()
        title = escape(c.title)
        nav = ""
        if c.html.navbar_show:
            links = "".join(
                f'<li class="nav-item"><a class="nav-link" href="#{s.anchor_id}">{s.name}</a></li>' for s in sections
            )
            nav = (
                '<a class="anchor-pos" id="top"></a><nav class="navbar navbar-expand-lg bg-body-tertiary sticky-top">'
                '<div class="container-fluid"><div class="navbar-header">'
                f'<a class="navbar-brand anchor" href="#top">{title}</a></div>'
                '<a class="navbar-toggler" type="button" data-bs-toggle="offcanvas" data-bs-target="#navbar" '
                'aria-controls="navbar" aria-label="Toggle navigation"><span class="navbar-toggler-icon"></span></a>'
                '<div id="navbar" class="offcanvas offcanvas-end" aria-labelledby="offcanvas-navbar-label">'
                f'<div class="offcanvas-header"><h5 id="offcanvas-navbar-label" class="offcanvas-title">{title}</h5>'
                '<a type="button" class="btn-close" data-bs-dismiss="offcanvas" aria-label="Close"></a></div>'
                '<div class="offcanvas-body"><ul class="navbar-nav navbar-nav-scroll justify-content-end flex-grow-1">'
                f"{links}</ul></div></div></div></nav>"
            )
        footer = f'Report generated by <a href="{REPO_URL}">lean-profiling</a>.'
        return (
            '<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">'
            f'<meta name="generator" content="lean-profiling v{__version__}">'
            f'<meta name="date" content="{d["date_start"]}">'
            f"<title>{re.sub(r'<[^>]+>', '', c.title)}</title>"
            f"<style>{_asset('bootstrap.min.css')}</style><style>{_style_css(c)}</style></head><body>"
            f'{nav}<div class="content">{body}</div>'
            '<footer><div class="container-fluid"><div class="row">'
            f'<p class="text-body-secondary text-center">{footer}</p></div></div></footer>'
            f"<script>{_asset('bootstrap.bundle.min.js')}</script><script>{_asset('script.js')}</script>"
            "</body></html>"
        )


def _asset(name: str) -> str:
    return resources.files("lean_profiling").joinpath("assets", name).read_text(encoding="utf-8")


def _style_css(config: Settings) -> str:
    css = _asset("style.css")
    nav = "html {\n    scroll-padding-top: var(--navbar-height);\n}" if config.html.navbar_show else ""
    colors = config.html.style.primary_colors
    bars = "\n".join(f".freq.table-{i} .bar {{\n    background-color: {colors[i]};\n}}" for i in range(min(2, len(colors))))
    return css.replace("/*NAV_SCROLL_PADDING*/", nav).replace("/*FREQ_BAR_COLORS*/", bars)
