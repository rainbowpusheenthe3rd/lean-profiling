"""A small SVG plotting engine that reproduces the look of ydata-profiling's matplotlib charts.

Geometry follows matplotlib (points, 72 per inch): the same figure sizes, subplot
parameters, ``tight_layout`` padding, tick locator (``MaxNLocator`` with steps
1/2/2.5/5/10) and ``ScalarFormatter`` rules, under ydata-profiling's style
(seaborn "white", 10pt Arial, no spines, no tick marks).
"""

import base64
import math
import struct
import zlib
from datetime import datetime, timedelta
from html import escape
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from lean_profiling._data import ARIAL_WIDTHS

FONT = "Arial, 'Liberation Sans', 'DejaVu Sans', sans-serif"
TEXT_COLOR = "#262626"  # matplotlib ".15"
TICK_PAD = 10.5  # xtick.major.pad (7) + tick size (3.5)
LAYOUT_PAD = 10.8  # tight_layout pad = 1.08 * font size
MINUS = "−"


# --- text metrics ---------------------------------------------------------------


def text_width(text: str, size: float, bold: bool = False) -> float:
    width = sum(ARIAL_WIDTHS.get(ch, 0.556) for ch in text) * size
    return width * (1.06 if bold else 1.0)


def text_height(size: float) -> float:
    return size


def text_descent(size: float) -> float:
    return 0.21 * size


def rotated_box(width: float, height: float, rotation: float) -> Tuple[float, float]:
    r = math.radians(rotation)
    c, s = abs(math.cos(r)), abs(math.sin(r))
    return width * c + height * s, width * s + height * c


def xml_text(text: str) -> str:
    """Escape for SVG, with non-ASCII as character references (safe in any page encoding)."""
    return escape(text).encode("ascii", "xmlcharrefreplace").decode("ascii")


class Label:
    """A text item positioned by its (rotated) bounding box, like matplotlib Text."""

    def __init__(self, text: str, x: float, y: float, size: float, rotation: float = 0, ha: str = "center",
                 va: str = "center", color: str = TEXT_COLOR, bold: bool = False, family: str = FONT):
        self.text, self.size, self.rotation, self.color, self.bold, self.family = text, size, rotation, color, bold, family
        lines = text.split("\n")
        self.lines = lines
        w = max(text_width(line, size, bold) for line in lines)
        h = text_height(size) * (1 + 1.2 * (len(lines) - 1))
        self.w, self.h = w, h
        bw, bh = rotated_box(w, h, rotation)
        if ha == "left":
            cx = x + bw / 2
        elif ha == "right":
            cx = x - bw / 2
        else:
            cx = x
        if va == "top":
            cy = y + bh / 2
        elif va == "bottom":
            cy = y - bh / 2
        elif va == "center_baseline":
            cy = y + text_descent(size) / 2
        else:
            cy = y
        self.cx, self.cy, self.bw, self.bh = cx, cy, bw, bh
        self.is_offset = False

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        return (self.cx - self.bw / 2, self.cy - self.bh / 2, self.cx + self.bw / 2, self.cy + self.bh / 2)

    def svg(self) -> str:
        line_step = 1.2 * self.size
        first = self.cy - (len(self.lines) - 1) * line_step / 2 + (text_height(self.size) / 2 - text_descent(self.size))
        weight = ' font-weight="bold"' if self.bold else ""
        rot = f' transform="rotate({-self.rotation:g} {self.cx:.3f} {self.cy:.3f})"' if self.rotation else ""
        tspans = "".join(
            f'<tspan x="{self.cx:.3f}" y="{first + i * line_step:.3f}">{xml_text(line)}</tspan>'
            for i, line in enumerate(self.lines)
        )
        return (
            f'<text text-anchor="middle" font-family="{self.family}" font-size="{self.size:g}" '
            f'fill="{self.color}"{weight}{rot}>{tspans}</text>'
        )


# --- ticks (matplotlib.ticker) ----------------------------------------------------------

_STEPS = np.array([0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10, 20])


def nonsingular(vmin: float, vmax: float, expander: float = 0.05, tiny: float = 1e-15) -> Tuple[float, float]:
    if not (math.isfinite(vmin) and math.isfinite(vmax)):
        return -expander, expander
    if vmax < vmin:
        vmin, vmax = vmax, vmin
    maxabs = max(abs(vmin), abs(vmax))
    if maxabs < 1e6 / tiny * np.finfo(float).tiny:
        return -expander, expander
    if vmax - vmin <= maxabs * tiny:
        if vmax == 0 and vmin == 0:
            return -expander, expander
        return vmin - expander * abs(vmin), vmax + expander * abs(vmax)
    return vmin, vmax


def _scale_range(vmin: float, vmax: float, n: int) -> Tuple[float, float]:
    dv = abs(vmax - vmin)
    meanv = (vmax + vmin) / 2
    offset = 0.0 if abs(meanv) / dv < 100 else math.copysign(10 ** (math.log10(abs(meanv)) // 1), meanv)
    return 10 ** (math.log10(dv / n) // 1), offset


def _closeto(ms: float, edge: float, step: float, offset: float) -> bool:
    if offset > 0:
        digits = math.log10(offset / step)
        tol = min(0.4999, max(1e-10, 10 ** (digits - 12)))
    else:
        tol = 1e-10
    return abs(ms - edge) < tol


def max_n_ticks(vmin: float, vmax: float, nbins: int) -> np.ndarray:
    """MaxNLocator(nbins, steps=[1, 2, 2.5, 5, 10]).tick_values."""
    vmin, vmax = nonsingular(vmin, vmax, expander=1e-13, tiny=1e-14)
    scale, offset = _scale_range(vmin, vmax, nbins)
    _vmin, _vmax = vmin - offset, vmax - offset
    steps = _STEPS * scale
    raw_step = (_vmax - _vmin) / nbins
    istep = int(np.nonzero(steps >= raw_step)[0][0])
    ticks = np.array([])
    for step in steps[: istep + 1][::-1]:
        best_vmin = (_vmin // step) * step
        d, m = divmod(_vmin - best_vmin, step)
        low = d + 1 if _closeto(m / step, 1, step, abs(offset)) else d
        d, m = divmod(_vmax - best_vmin, step)
        high = d if _closeto(m / step, 0, step, abs(offset)) else d + 1
        ticks = np.arange(low, high + 1) * step + best_vmin
        if ((ticks <= _vmax) & (ticks >= _vmin)).sum() >= 2:
            break
    return ticks + offset


def tick_space(length: float, horizontal: bool) -> int:
    """Axis.get_tick_space, with the rcParams tick label size of 10pt."""
    return int(math.floor(length / (10 * (3 if horizontal else 2))))


def auto_ticks(lo: float, hi: float, length: float, horizontal: bool) -> np.ndarray:
    nbins = int(np.clip(tick_space(length, horizontal), 1, 9))
    return max_n_ticks(min(lo, hi), max(lo, hi), nbins)


def _fix_minus(s: str) -> str:
    return s.replace("-", MINUS)


def scalar_format(locs: Sequence[float], view: Tuple[float, float]) -> Tuple[List[str], str]:
    """ScalarFormatter: tick labels plus the offset/exponent text shown at the axis end."""
    locs = np.asarray(locs, dtype=float)
    lo, hi = min(view), max(view)
    visible = locs[(locs >= lo) & (locs <= hi)] if len(locs) else locs
    # Offset (axes.formatter.offset_threshold = 4).
    offset = 0.0
    if len(visible):
        lmin, lmax = visible.min(), visible.max()
        if not (lmin == lmax or lmin <= 0 <= lmax):
            abs_min, abs_max = sorted([abs(float(lmin)), abs(float(lmax))])
            sign = math.copysign(1, lmin)
            oom_max = math.ceil(math.log10(abs_max))
            oom = oom_max
            while oom > -400 and abs_min // 10**oom == abs_max // 10**oom:
                oom -= 1
            oom += 1
            if (abs_max - abs_min) / 10**oom <= 1e-2:
                # Straddling a multiple of a large power of ten (relative to the span).
                oom = oom_max
                while oom > -400 and not abs_max - abs_min > 10**oom:
                    oom -= 1
                oom += 1
            offset = sign * (abs_max // 10**oom) * 10**oom if abs_max // 10**oom >= 10**3 else 0.0
    # Order of magnitude (axes.formatter.limits = -5, 6).
    order = 0
    if len(visible):
        val = np.abs(visible - offset).max()
        if val != 0:
            oom = math.floor(math.log10(val))
            if oom <= -5 or oom >= 6:
                order = oom
    scaled = (locs - offset) / 10.0**order
    if len(scaled) < 2:
        scaled_all = np.concatenate([scaled, (np.array(view, dtype=float) - offset) / 10.0**order])
    else:
        scaled_all = scaled
    loc_range = float(np.ptp(scaled_all)) or float(np.max(np.abs(scaled_all))) or 1.0
    loc_range_oom = int(math.floor(math.log10(loc_range)))
    sigfigs = max(0, 3 - loc_range_oom)
    thresh = 1e-3 * 10**loc_range_oom
    while sigfigs >= 0:
        if np.abs(scaled - np.round(scaled, decimals=sigfigs)).max(initial=0) < thresh:
            sigfigs -= 1
        else:
            break
    sigfigs += 1
    labels = [_fix_minus(f"%1.{sigfigs}f" % v) for v in scaled]
    offset_text = ""
    if order:
        offset_text += f"1e{order}"
    if offset:
        e = math.floor(math.log10(abs(offset)))
        s = round(offset / 10**e, 10)
        significand = "%d" % s if s % 1 == 0 else ("%1.10f" % s).rstrip("0")
        offset_text += ("+" if offset > 0 else "") + (f"{significand}e{e}" if e else significand)
    return labels, _fix_minus(offset_text)


def timestamp_label(value: float) -> str:
    """ydata-profiling's date tick format (local time, as datetime.fromtimestamp)."""
    if value >= 0:
        moment = datetime.fromtimestamp(value)
    else:
        moment = datetime(1970, 1, 1) + timedelta(seconds=int(value))
    return moment.strftime("%Y-%m-%d %H:%M:%S")


# --- axes ----------------------------------------------------------------------------------


class Axis:
    def __init__(self, lim: Tuple[float, float], horizontal: bool, side: Optional[str] = None,
                 ticks: Optional[Sequence[float]] = None, labels: Optional[Sequence[str]] = None,
                 formatter: Optional[Callable[[float], str]] = None, fontsize: float = 10, rotation: float = 0,
                 ha: Optional[str] = None, va: Optional[str] = None, label: Optional[str] = None,
                 labelsize: float = 11, visible: bool = True, tick_marks: bool = False):
        self.lim = lim
        self.horizontal = horizontal
        self.side = side or ("bottom" if horizontal else "left")
        self.fixed_ticks, self.fixed_labels = ticks, labels
        self.formatter = formatter
        self.fontsize, self.rotation = fontsize, rotation
        self.ha = ha or ("center" if horizontal else ("right" if self.side == "left" else "left"))
        self.va = va or (("top" if self.side == "bottom" else "bottom") if horizontal else "center_baseline")
        self.label, self.labelsize = label, labelsize
        self.visible = visible
        self.tick_marks = tick_marks

    def ticks(self, length: float) -> Tuple[np.ndarray, List[str], str]:
        if self.fixed_ticks is not None:
            locs = np.asarray(self.fixed_ticks, dtype=float)
            labels = list(self.fixed_labels) if self.fixed_labels is not None else [str(v) for v in locs]
            return locs, labels, ""
        locs = auto_ticks(self.lim[0], self.lim[1], length, self.horizontal)
        if self.formatter is not None:
            return locs, [self.formatter(v) for v in locs], ""
        labels, offset = scalar_format(locs, self.lim)
        return locs, labels, offset


class Axes:
    """One set of axes: data limits, two axes, and a rectangle on the figure."""

    def __init__(self, xaxis: Axis, yaxis: Axis, extra_axes: Sequence[Axis] = ()):
        self.xaxis, self.yaxis = xaxis, yaxis
        self.extra_axes = list(extra_axes)
        self.rect = (0.0, 0.0, 1.0, 1.0)  # left, top, right, bottom in points

    def set_rect(self, rect: Tuple[float, float, float, float]) -> None:
        self.rect = rect

    def x(self, value):
        left, _, right, _ = self.rect
        lo, hi = self.xaxis.lim
        return left + (np.asarray(value, dtype=float) - lo) / (hi - lo) * (right - left)

    def y(self, value):
        _, top, _, bottom = self.rect
        lo, hi = self.yaxis.lim
        return bottom - (np.asarray(value, dtype=float) - lo) / (hi - lo) * (bottom - top)

    def _axis_labels(self, axis: Axis) -> List[Label]:
        if not axis.visible:
            return []
        left, top, right, bottom = self.rect
        lo, hi = sorted(axis.lim)
        length = (right - left) if axis.horizontal else (bottom - top)
        locs, labels, offset_text = axis.ticks(length)
        out = []
        eps = 1e-9 * max(1.0, abs(hi - lo))
        for loc, text in zip(locs, labels):
            if loc < lo - eps or loc > hi + eps or text == "":
                continue
            if axis.horizontal:
                px = float(self.x(loc))
                py = bottom + TICK_PAD if axis.side == "bottom" else top - TICK_PAD
            else:
                py = float(self.y(loc))
                px = left - TICK_PAD if axis.side == "left" else right + TICK_PAD
            out.append(Label(text, px, py, axis.fontsize, axis.rotation, axis.ha, axis.va))
        if offset_text:
            # Axis._update_offset_text_position: OFFSETTEXTPAD (3pt) beyond the tick labels.
            if axis.horizontal:
                below = max([lbl.bbox[3] for lbl in out] + [bottom + TICK_PAD])
                offset_label = Label(offset_text, right, below + 3, axis.fontsize, ha="right", va="top")
            else:
                offset_label = Label(offset_text, left, top - 3, axis.fontsize, ha="left", va="bottom")
            offset_label.is_offset = True
            out.append(offset_label)
        return out

    def tick_label_items(self) -> List[Label]:
        items: List[Label] = []
        for axis in [self.xaxis, self.yaxis, *self.extra_axes]:
            items += self._axis_labels(axis)
        return items

    def axis_title_items(self, tick_items: List[Label]) -> List[Label]:
        left, top, right, bottom = self.rect
        out = []
        if self.xaxis.label:
            below = max([lbl.bbox[3] for lbl in tick_items if lbl in self._x_items and not lbl.is_offset] + [bottom])
            out.append(Label(self.xaxis.label, (left + right) / 2, below + 4, self.xaxis.labelsize, va="top"))
        if self.yaxis.label:
            leftmost = min([lbl.bbox[0] for lbl in tick_items if lbl in self._y_items and not lbl.is_offset] + [left])
            out.append(Label(self.yaxis.label, leftmost - 4, (top + bottom) / 2, self.yaxis.labelsize, 90, ha="right"))
        return out

    def all_labels(self) -> List[Label]:
        self._x_items = self._axis_labels(self.xaxis)
        self._y_items = self._axis_labels(self.yaxis)
        ticks = self._x_items + self._y_items
        for axis in self.extra_axes:
            ticks += self._axis_labels(axis)
        return ticks + self.axis_title_items(ticks)

    def tick_marks_svg(self) -> str:
        parts = []
        left, top, right, bottom = self.rect
        for axis in [self.xaxis, self.yaxis, *self.extra_axes]:
            if not (axis.visible and axis.tick_marks):
                continue
            length = (right - left) if axis.horizontal else (bottom - top)
            locs, _, _ = axis.ticks(length)
            lo, hi = sorted(axis.lim)
            for loc in locs:
                if loc < lo - 1e-9 or loc > hi + 1e-9:
                    continue
                if axis.horizontal:
                    px = float(self.x(loc))
                    y0 = bottom if axis.side == "bottom" else top
                    y1 = y0 + 3.5 if axis.side == "bottom" else y0 - 3.5
                    parts.append(line(px, y0, px, y1, TEXT_COLOR, 1))
                else:
                    py = float(self.y(loc))
                    x0 = left if axis.side == "left" else right
                    x1 = x0 - 3.5 if axis.side == "left" else x0 + 3.5
                    parts.append(line(x0, py, x1, py, TEXT_COLOR, 1))
        return "".join(parts)


def subplot_rect(width: float, height: float, left=0.125, right=0.9, bottom=0.11, top=0.88):
    return (width * left, height * (1 - top), width * right, height * (1 - bottom))


def tight_layout(axes: Axes, width: float, height: float, pad: float = LAYOUT_PAD) -> None:
    """Figure.tight_layout for a single axes: measure labels at the default position, then fit."""
    axes.set_rect(subplot_rect(width, height))
    left, top, right, bottom = axes.rect
    items = axes.all_labels()
    x0 = min([i.bbox[0] for i in items] + [left])
    y0 = min([i.bbox[1] for i in items] + [top])
    x1 = max([i.bbox[2] for i in items] + [right])
    y1 = max([i.bbox[3] for i in items] + [bottom])
    axes.set_rect((pad + (left - x0), pad + (top - y0), width - pad - (x1 - right), height - pad - (y1 - bottom)))


# --- primitives -------------------------------------------------------------------------------


def _f(v: float) -> str:
    return f"{v:.3f}".rstrip("0").rstrip(".")


def rect(x0: float, y0: float, x1: float, y1: float, fill: str, stroke: Optional[str] = None,
         stroke_width: float = 0.3, extra: str = "") -> str:
    x, y = min(x0, x1), min(y0, y1)
    w, h = abs(x1 - x0), abs(y1 - y0)
    s = f' stroke="{stroke}" stroke-width="{stroke_width:g}"' if stroke else ""
    return f'<rect x="{_f(x)}" y="{_f(y)}" width="{_f(w)}" height="{_f(h)}" fill="{fill}"{s}{extra}/>'


def line(x0: float, y0: float, x1: float, y1: float, color: str, width: float) -> str:
    return f'<line x1="{_f(x0)}" y1="{_f(y0)}" x2="{_f(x1)}" y2="{_f(y1)}" stroke="{color}" stroke-width="{width:g}"/>'


def polygon(points: Sequence[Tuple[float, float]], fill: str, stroke: Optional[str] = None, width: float = 1.0) -> str:
    pts = " ".join(f"{_f(x)},{_f(y)}" for x, y in points)
    s = f' stroke="{stroke}" stroke-width="{width:g}"' if stroke else ""
    return f'<polygon points="{pts}" fill="{fill}"{s}/>'


def svg_document(width: float, height: float, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_f(width)}pt" height="{_f(height)}pt" '
        f'viewBox="0 0 {_f(width)} {_f(height)}" version="1.1">'
        f'{rect(0, 0, width, height, "#ffffff")}{body}</svg>'
    )


def labels_svg(items: Sequence[Label]) -> str:
    return "".join(item.svg() for item in items)


# --- colours ------------------------------------------------------------------------------------


def hex_to_rgb(color: str) -> Tuple[float, float, float]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore[return-value]


def rgb_to_hex(rgb: Sequence[float]) -> str:
    return "#" + "".join(f"{int(round(max(0.0, min(1.0, c)) * 255)):02x}" for c in rgb[:3])


def lut_color(lut: Sequence[str], value: float) -> str:
    """Matplotlib colormap lookup for value in [0, 1] (N=256)."""
    if not math.isfinite(value):
        return "#000000"
    idx = int(min(max(value, 0.0), 1.0) * 256)
    return lut[min(idx, 255)]


def blend(c0: str, c1: str, t: float) -> str:
    a, b = hex_to_rgb(c0), hex_to_rgb(c1)
    return rgb_to_hex([x + (y - x) * t for x, y in zip(a, b)])


_GRADIENT_ID = [0]


def colorbar_svg(x0: float, y0: float, x1: float, y1: float, lut: Sequence[str], stops: int = 64) -> str:
    """A vertical colour bar, top = max."""
    _GRADIENT_ID[0] += 1
    gid = f"cbar{_GRADIENT_ID[0]}"
    stop_tags = "".join(
        f'<stop offset="{i / (stops - 1):.4f}" stop-color="{lut_color(lut, 1 - i / (stops - 1))}"/>'
        for i in range(stops)
    )
    return (
        f'<defs><linearGradient id="{gid}" x1="0" y1="0" x2="0" y2="1">{stop_tags}</linearGradient></defs>'
        f'<rect x="{_f(x0)}" y="{_f(y0)}" width="{_f(x1 - x0)}" height="{_f(y1 - y0)}" fill="url(#{gid})"/>'
    )


def png_rgb(pixels: np.ndarray) -> bytes:
    """Encode an (height, width, 3) uint8 array as a PNG, with only the standard library."""
    height, width, _ = pixels.shape
    raw = np.concatenate([np.zeros((height, 1), dtype=np.uint8), pixels.reshape(height, width * 3)], axis=1)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    signature = bytes([0x89]) + b"PNG" + bytes([0x0D, 0x0A, 0x1A, 0x0A])
    return signature + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw.tobytes(), 9)) + chunk(b"IEND", b"")


def pixel_image(pixels: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> str:
    """An image drawn one cell per data point, like matplotlib's imshow(interpolation="none") in SVG."""
    height, width, _ = pixels.shape
    data = base64.b64encode(png_rgb(pixels)).decode("ascii")
    sx, sy = (x1 - x0) / width, (y1 - y0) / height
    return (
        f'<image transform="matrix({sx:.6f} 0 0 {sy:.6f} {x0:.3f} {y0:.3f})" '
        f'style="image-rendering:crisp-edges;image-rendering:pixelated" width="{width}" height="{height}" '
        f'href="data:image/png;base64,{data}"/>'
    )
