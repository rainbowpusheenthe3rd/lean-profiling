"""HTML components: a port of ydata-profiling's Jinja templates (``report/presentation/flavours/html``)."""

import re
from html import escape as _html_escape
from typing import Any, List, Optional, Sequence


def escape(value: Any) -> str:
    """markupsafe.escape."""
    return (
        _html_escape(str(value), quote=False).replace('"', "&#34;").replace("'", "&#39;")
    )


def fmt_badge(value: str) -> str:
    return re.sub(r"\((\d+)\)", r'<span class="badge text-bg-secondary align-text-top">\1</span>', value)


class Renderable:
    def __init__(self, name: Optional[str] = None, anchor_id: Optional[str] = None, classes: Optional[str] = None):
        self.name, self.anchor_id, self.classes = name, anchor_id, classes

    def render(self) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class HTML(Renderable):
    def __init__(self, content: str, **kwargs):
        super().__init__(**kwargs)
        self.content = content

    def render(self) -> str:
        return self.content


class Table(Renderable):
    def __init__(self, rows: List[dict], name: Optional[str] = None, caption: Optional[str] = None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.rows, self.caption = rows, caption

    def render(self) -> str:
        out = ['<div class="table-responsive">']
        if self.name:
            out.append(f'<p class="h4 item-header">{self.name}</p>')
        if self.rows:
            out.append('<table class="table table-striped"><tbody>')
            for row in self.rows:
                cls = ' class="alert-info"' if row.get("alert") else ""
                hint = f" {row['hint']}" if "hint" in row else ""
                out.append(f'<tr{cls}><th>{row["name"]}</th><td style="white-space: nowrap;">{row["value"]}{hint}</td></tr>')
            out.append("</tbody></table>")
        if self.caption:
            out.append(f'<div class="caption text-center text-body-secondary">{self.caption}</div>')
        out.append("</div>")
        return "".join(out)


def _bar_percent(width: float) -> str:
    return f"{width * 100:2.1f}%"


class FrequencyTable(Renderable):
    def __init__(self, rows: List[dict], redact: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.rows, self.redact = rows, redact

    def render(self) -> str:
        from lean_profiling.formatters import fmt, fmt_percent

        out = [
            '<div class="table-responsive"><table class="freq table table-hover table-striped table-0">'
            "<thead><tr><td>Value</td><td>Count</td><td>Frequency (%)</td></tr></thead><tbody>"
        ]
        if not self.rows:
            out.append('<tr><td colspan="3">No values found.</td></tr>')
        for row in self.rows:
            label = fmt(row["label"])
            inside = fmt_percent(row["percentage"]) if row["width"] > 0.40 else "&nbsp;"
            after = fmt_percent(row["percentage"]) if row["width"] <= 0.40 else ""
            out.append(
                f'<tr class="{row["extra_class"]}"><td title="{label}">{label}</td><td>{row["count"]}</td>'
                f'<td><div class="bar" style="width:{_bar_percent(row["width"])}">{inside}</div>{after}</td></tr>'
            )
        out.append("</tbody></table></div>")
        return "".join(out)


class FrequencyTableSmall(Renderable):
    def __init__(self, rows: List[dict], varid: Any, redact: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.rows, self.varid, self.redact = rows, varid, redact

    def render(self) -> str:
        from lean_profiling.formatters import fmt

        out = [f'<div class="row" id="minifreqtable{self.varid}"><div class="table-responsive">'
               '<table class="mini freq table-0">']
        for row in self.rows:
            label = "" if self.redact else fmt(row["label"])
            inside = f"{row['count']}&nbsp;" if row["width"] > 0.10 else "&nbsp;"
            after = str(row["count"]) if row["width"] <= 0.10 else ""
            out.append(
                f'<tr class="{row["extra_class"]}"><th width="10%">{label}</th><td width="10%">'
                f'<div class="bar" style="width:{_bar_percent(row["width"])}" data-bs-toggle="tooltip" '
                f'data-bs-placement="center" data-bs-html="true" data-bs-delay="500">{inside}</div>{after}</td></tr>'
            )
        out.append("</table></div></div>")
        return "".join(out)


class Image(Renderable):
    def __init__(self, image: str, alt: str, caption: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.image, self.alt, self.caption = image, alt, caption

    def render(self) -> str:
        out = self.image.replace("<svg ", '<svg class="img-responsive center-img" ', 1)
        if self.caption:
            out += f'<div class="caption text-center text-body-secondary">{self.caption}</div>'
        return out


def image_or_empty(image: Optional[str], alt: str, caption: Optional[str] = None, name: Optional[str] = None,
                   anchor_id: Optional[str] = None) -> Renderable:
    if image is None:
        return Container([], "grid", name=name or f"{alt}_empty", anchor_id=anchor_id)
    return Image(image, alt=alt, caption=caption, name=name, anchor_id=anchor_id)


class VariableInfo(Renderable):
    def __init__(self, anchor_id: Any, var_name: str, var_type: str, alerts: List[str], description: str, **kwargs):
        super().__init__(anchor_id=anchor_id, **kwargs)
        self.var_name, self.var_type, self.alerts, self.description = var_name, var_type, alerts, description

    def render(self) -> str:
        name = escape(self.var_name)
        out = (
            f'<p class="item-header h4" title="{name}"><a href="#pp_var_{self.anchor_id}">{name}</a><br/>'
            f'<span class="fs-6 text-body-secondary">{self.var_type}</span></p>'
        )
        if self.alerts:
            out += "<p>" + "".join(f"{a}&nbsp" for a in self.alerts) + "</p>"
        return out + f'<p class="variable-description">{self.description}</p>'


class Alerts(Renderable):
    def __init__(self, alerts: list, **kwargs):
        super().__init__(**kwargs)
        self.alerts = alerts

    def render(self) -> str:
        from lean_profiling.formatters import alert_html

        rows = "".join(
            f'<tr><td>{alert_html(a)}</td><td><span class="badge text-bg-{a.style}">{a.type_name}</span></td></tr>'
            for a in self.alerts
            if a.name != "REJECTED"
        )
        return f'<div class="col-sm-12"><table class="table table-striped"><p class="h4 item-header">Alerts</p>{rows}</table></div>'


class DataFrameTable(Renderable):
    """Sample, duplicate rows and correlation tables (pandas ``DataFrame.to_html``)."""

    def __init__(self, html: str, container_id: str, caption: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.html, self.container_id, self.caption = html, container_id, caption

    def render(self) -> str:
        caption = f'<div class="caption text-center text-body-secondary">{self.caption}</div>' if self.caption else ""
        return (
            f'<div id="{self.container_id}" class="col-sm-12"><div class="table-responsive">{self.html}{caption}'
            "</div></div>"
        )


class Variable(Renderable):
    def __init__(self, top: Renderable, bottom: Optional[Renderable], anchor_id: Any, name: str, ignore: bool):
        super().__init__(name=name, anchor_id=anchor_id)
        self.top, self.bottom, self.ignore = top, bottom, ignore

    def render(self) -> str:
        bottom = ""
        if self.bottom is not None:
            aid = self.anchor_id
            bottom = (
                '<div class="col-sm-12 text-end"><button class="btn btn-light" data-bs-toggle="collapse" '
                f'data-bs-target="#bottom-{aid}, #minifreqtable{aid}" aria-expanded="true" '
                f'aria-controls="#bottom-{aid}, #minifreqtable{aid}">More details</button></div>'
                f'<div id="bottom-{aid}" class="collapse">{self.bottom.render()}</div>'
            )
        ignore = "ignore" if self.ignore else ""
        return f'<div id="pp_var_{self.anchor_id}" class="variable {ignore}">{self.top.render()}{bottom}</div>'


class Dropdown(Renderable):
    def __init__(self, name: str, element_id: str, items: Sequence[str], item: Renderable, anchor_id: str):
        super().__init__(name=name, anchor_id=anchor_id)
        self.element_id, self.items, self.item = element_id, items, item

    def render(self) -> str:
        options = "".join(f'<option value="{escape(i)}">{escape(i)}</option>' for i in self.items)
        return (
            f'<select name="{self.name}" id="{self.element_id}" class="form-select">'
            f'<option value="">Select Columns</option>{options}</select>{self.item.render()}'
        )


class Container(Renderable):
    def __init__(self, items: Sequence[Renderable], sequence_type: str, nested: bool = False, batch_size: int = 1,
                 titles: bool = True, subtitles: bool = False, full_width: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.items, self.sequence_type, self.nested = list(items), sequence_type, nested
        self.batch_size, self.titles, self.subtitles, self.full_width = batch_size, titles, subtitles, full_width

    @staticmethod
    def _cls(item: Renderable) -> str:
        return f" {item.classes}" if item.classes else ""

    def render(self) -> str:
        kind = self.sequence_type
        if kind in ("list", "accordion"):
            return "".join(f'<div class="row item">{i.render()}</div>' for i in self.items)
        if kind == "named_list":
            return "".join(
                f'<div class="row named-list-item"><h4 class="item-header">{i.name}</h4>{i.render()}</div>'
                for i in self.items
            )
        if kind in ("tabs", "overview_tabs"):
            return self._tabs()
        if kind == "select":
            return self._select()
        if kind == "sections":
            return self._sections()
        if kind == "grid":
            return self._grid()
        if kind == "batch_grid":
            return self._batch_grid()
        raise ValueError(f"Template not understood: {kind}")

    def _tabs(self) -> str:
        if not self.items:
            return ""
        aid = self.anchor_id
        nav, panes = [], []
        for idx, tab in enumerate(self.items):
            first = idx == 0
            nav.append(
                f'<li class="nav-item" role="presentation"><button class="nav-link {"active" if first else ""}" '
                f'id="tab-{aid}-{tab.anchor_id}" data-bs-toggle="tab" data-bs-target="#tab-pane-{aid}-{tab.anchor_id}" '
                f'type="button" role="tab" aria-controls="tab-pane-{aid}-{tab.anchor_id}" '
                f'aria-selected="{"true" if first else "false"}">{fmt_badge(str(tab.name))}</button></li>'
            )
            panes.append(
                f'<div class="tab-pane fade {"show active" if first else ""}" id="tab-pane-{aid}-{tab.anchor_id}" '
                f'role="tabpanel" aria-labelledby="tab-pane-{aid}-{tab.anchor_id}"><div class="row item">'
                f"{tab.render()}</div></div>"
            )
        cls = f" {self.classes}" if self.classes else " "
        return (
            f'<div class="row item{cls}"><ul class="nav nav-tabs tab-nav" role="tablist">{"".join(nav)}</ul>'
            f'<div class="tab-content">{"".join(panes)}</div></div>'
        )

    def _select(self) -> str:
        if not self.items:
            return ""
        c1, c2, border = (3, 9, ' style="border: 0px;"') if self.nested else (2, 10, "")
        aid = self.anchor_id
        options, panes = [], []
        for idx, tab in enumerate(self.items):
            first = idx == 0
            options.append(
                f'<option class="tab-select-option {"active" if first else ""}" data-bs-toggle="tab" '
                f'data-bs-target="#tab-pane-{aid}-{tab.anchor_id}" role="tab" aria-controls="tab-pane-{aid}-{tab.anchor_id}" '
                f'aria-selected="{"true" if first else "false"}">{fmt_badge(str(tab.name))}</option>'
            )
            panes.append(
                f'<div class="tab-pane col-sm-12 {"show active" if first else ""}" id="tab-pane-{aid}-{tab.anchor_id}" '
                f'role="tabpanel">{tab.render()}</div>'
            )
        return (
            f'<div class="row {self.classes or "variable"}"{border}><div class="col-sm-{c1}">'
            f'<select class="form-select tab-select" role="tablist" id="select-" multiple>{"".join(options)}'
            f'</select></div><div class="tab-content col-sm-{c2}">{"".join(panes)}</div></div>'
        )

    def _sections(self) -> str:
        out = [f'<div class="{"container-fluid" if self.full_width else "container"}">']
        for section in self.items:
            html = section.render()
            if html:
                out.append(
                    f'<div id="{section.anchor_id}" class="section-header"><h1 class="section-name">{section.name}</h1>'
                    f'</div><div class="section-items">{html}</div>'
                )
        out.append("</div>")
        return "".join(out)

    def _grid(self) -> str:
        items = self.items
        out = ['<div class="row sub-item">']
        if items and isinstance(items[0], VariableInfo):
            out.append(f'<div class="col-sm-12">{items[0].render()}</div>')
            rest = items[1:]
            width = 4 if len(rest) == 3 else 6
            out += [f'<div class="col-sm-{width}{self._cls(i)}">{i.render()}</div>' for i in rest]
        else:
            n = len(items)
            for idx, item in enumerate(items):
                last = idx == n - 1
                width = 12 if n == 1 else 6 if (last and n == 3) or n == 2 else 3
                out.append(f'<div class="col-sm-{width}{self._cls(item)}">{item.render()}</div>')
        out.append("</div>")
        return "".join(out)

    def _batch_grid(self) -> str:
        size = max(1, self.batch_size)
        out = []
        for start in range(0, len(self.items), size):
            out.append('<div class="row sub-item">')
            for item in self.items[start : start + size]:
                header = ""
                if item.name is not None:
                    if self.subtitles:
                        header = f'<h5 class="item-header">{item.name}</h5>'
                    elif self.titles:
                        header = f'<h4 class="item-header">{item.name}</h4>'
                out.append(f'<div class="col-sm-{int(12 / size)}{self._cls(item)}">{header}{item.render()}</div>')
            out.append("</div>")
        return "".join(out)

