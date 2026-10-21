"""The report's charts, matching ydata-profiling's ``visualisation`` module."""

import math
import random
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from lean_profiling import svg
from lean_profiling._data import LIGHT_PALETTE_GRAY, MONO_WIDTH, RDBU, VIRIDIS
from lean_profiling.config import Settings
from lean_profiling.svg import Axes, Axis, Label

# matplotlib's default colour cycle ("tab10"), used for category frequency plots.
TAB10 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
WHITE = "#ffffff"
TEXT_DARK = svg.TEXT_COLOR


def _primary(config: Settings) -> str:
    return config.html.style.primary_colors[0]


def _margins(lo: float, hi: float, margin: float = 0.05):
    lo, hi = svg.nonsingular(lo, hi)
    span = hi - lo
    return lo - margin * span, hi + margin * span


def _bar_ylim(top: float):
    if top <= 0:
        return svg.nonsingular(0.0, 0.0)
    return 0.0, top * 1.05


# --- histograms (plot.histogram / plot.mini_histogram) --------------------------------


def _valid_hist(counts, edges) -> bool:
    if counts is None or edges is None or len(counts) == 0:
        return False
    try:
        counts = np.asarray(counts, dtype=float)
        edges = np.asarray(edges, dtype=float)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(counts).any()) and len(edges) >= 2 and bool(np.isfinite(edges).all())


def histogram(config: Settings, counts, edges, date: bool = False, mini: bool = False) -> Optional[str]:
    if not _valid_hist(counts, edges):
        return None
    counts = np.asarray(counts, dtype=float)
    edges = np.asarray(edges, dtype=float)
    width, height = (216.0, 162.0) if mini else (504.0, 216.0)
    show_x = config.plot.histogram.x_axis_labels
    xaxis = Axis(
        _margins(edges[0], edges[-1]),
        True,
        formatter=svg.timestamp_label if date else None,
        fontsize=(6 if date else 8) if mini else 10,
        rotation=90 if date else 45,
        visible=show_x,
    )
    yaxis = Axis(_bar_ylim(float(np.nanmax(counts))), False, label=None if mini else "Frequency", visible=not mini)
    ax = Axes(xaxis, yaxis)
    svg.tight_layout(ax, width, height)
    color = _primary(config)
    bars = [
        svg.rect(float(ax.x(e0)), float(ax.y(c)), float(ax.x(e1)), float(ax.y(0)), color, WHITE)
        for e0, e1, c in zip(edges[:-1], edges[1:], counts)
        if np.isfinite(c)
    ]
    return svg.svg_document(width, height, "".join(bars) + svg.labels_svg(ax.all_labels()))


# --- category frequency (plot.cat_frequency_plot, stacked horizontal bar) ---------------


def cat_frequency_plot(config: Settings, data: pd.Series) -> str:
    colors = list(config.plot.cat_freq.colors or TAB10)
    if len(colors) < len(data):
        colors = (colors * (len(data) // len(colors) + 1))[: len(data)]
    labels = data.index.values.astype(str)
    total = float(np.sum(data))

    # Default subplot of a 7 x 2 inch figure, axis off; xlim (0, total), ylim (0.4, 1.6).
    left, top, right, bottom = svg.subplot_rect(504, 144)
    axes_h = bottom - top

    def ypix(v: float) -> float:
        return bottom - (v - 0.4) / 1.2 * axes_h

    def xpix(v: float) -> float:
        return left + v / total * (right - left) if total else left

    parts: List[str] = []
    texts: List[Label] = []
    start = 0.0
    for value, color in zip(data, colors):
        x0, x1 = xpix(start), xpix(start + value)
        parts.append(svg.rect(x0, ypix(1.5), x1, ypix(0.5), color, WHITE))
        r, g, b = svg.hex_to_rgb(color)
        pct = value / total * 100 if total else 0
        if pct > 8:
            texts.append(
                Label(f"{pct:.1f}%\n({value})", (x0 + x1) / 2, ypix(1.0), 14.4, bold=True,
                      color=WHITE if r * g * b < 0.5 else "#a9a9a9")
            )
        start += value

    # Legend below the axes: loc="upper left", bbox_to_anchor=(0, 0), fontsize="xx-large".
    fs = 17.28
    pad, spacing, handle_w, handle_h, text_pad = 0.4 * fs, 0.5 * fs, 2.0 * fs, 0.7 * fs, 0.8 * fs
    row_h = 0.9145 * fs
    leg_left, leg_top = left + 0.5 * fs, bottom + 0.5 * fs
    text_w = max((svg.text_width(lbl, fs) for lbl in labels), default=0)
    leg_w = 2 * pad + handle_w + text_pad + text_w
    leg_h = 2 * pad + len(labels) * row_h + (len(labels) - 1) * spacing
    legend = [
        f'<rect x="{leg_left:.3f}" y="{leg_top:.3f}" width="{leg_w:.3f}" height="{leg_h:.3f}" rx="{0.2 * fs:.3f}" '
        f'fill="#ffffff" fill-opacity="0.8" stroke="#cccccc" stroke-width="0.3"/>'
    ]
    for i, (lbl, color) in enumerate(zip(labels, colors)):
        cy = leg_top + pad + i * (row_h + spacing) + row_h / 2
        hx = leg_left + pad
        legend.append(svg.rect(hx, cy - handle_h / 2, hx + handle_w, cy + handle_h / 2, color, WHITE))
        texts.append(Label(lbl, hx + handle_w + text_pad, cy, fs, ha="left"))

    # bbox_inches="tight" with 0.1 inch padding.
    x0, y0 = min(left, leg_left), top
    x1, y1 = max(right, leg_left + leg_w), leg_top + leg_h
    width, height = (x1 - x0) + 14.4, (y1 - y0) + 14.4
    body = "".join(parts) + "".join(legend) + svg.labels_svg(texts)
    shifted = f'<g transform="translate({7.2 - x0:.3f} {7.2 - y0:.3f})">{body}</g>'
    return svg.svg_document(width, height, shifted)


# --- word cloud (plot.plot_word_cloud; layout after the wordcloud package) ----------------------
#
# wordcloud places each word at a random spot where its tight ink box is free of ink
# already drawn, so small words nest in the gaps of big ones. Glyph ink is modelled
# per character: x-height letters, ascenders/digits/capitals, and descenders.

MONO_FONT = "'Droid Sans Mono', 'DejaVu Sans Mono', Consolas, 'Courier New', monospace"
_ASCENDERS = set("bdfhklt0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ!?#$%&@/|()[]{}'\"ij\\")
_DESCENDERS = set("gjpqy,;()[]{}|@$")
_X_HEIGHT, _ASCENT, _DESCENT = 0.55, 0.74, 0.22


def _glyph_extent(ch: str):
    top = _ASCENT if ch in _ASCENDERS else (_X_HEIGHT if ch.strip() else 0.0)
    bottom = _DESCENT if ch in _DESCENDERS else 0.0
    return top, bottom


def _word_ink(word: str, size: int):
    """Tight ink box (width, height), baseline offset and per-character ink rectangles."""
    extents = [_glyph_extent(ch) for ch in word]
    top = max(e[0] for e in extents)
    bottom = max(e[1] for e in extents)
    advance = MONO_WIDTH * size
    width = max(1, int(math.ceil(advance * len(word))))
    height = max(1, int(math.ceil((top + bottom) * size)))
    baseline = top * size
    rects = []
    for i, (t, b) in enumerate(extents):
        if t == 0 and b == 0:
            continue
        rects.append((int(i * advance), int(baseline - t * size), int(math.ceil((i + 1) * advance)),
                      int(math.ceil(baseline + b * size))))
    return width, height, baseline, rects


def _free_spot(occupied: np.ndarray, w: int, h: int, rng: random.Random):
    rows, cols = occupied.shape
    if h >= rows or w >= cols:
        return None
    integral = np.zeros((rows + 1, cols + 1), dtype=np.int64)
    integral[1:, 1:] = occupied.cumsum(0).cumsum(1)
    area = integral[h:, w:] - integral[:-h, w:] - integral[h:, :-w] + integral[:-h, :-w]
    free = np.argwhere(area[: rows - h, : cols - w] == 0)
    if len(free) == 0:
        return None
    return tuple(free[rng.randint(0, len(free) - 1)])


def word_cloud(word_counts: pd.Series) -> str:
    width, height = 432.0, 288.0
    canvas_w, canvas_h, margin = 300, 200, 2
    words = [(str(w), float(c)) for w, c in word_counts.items() if str(w)][:200]
    if not words:
        return svg.svg_document(width, height, "")
    max_freq = words[0][1]
    words = [(w, c / max_freq) for w, c in words]

    def fits(word: str, size: int) -> bool:
        w, h, _, _ = _word_ink(word, size)
        return w + margin < canvas_w and h + margin < canvas_h

    if len(words) == 1:
        font_size = canvas_h
    else:
        # wordcloud sizes the run from a trial layout of the two most frequent words.
        sizes, size = [], canvas_h
        for i, (word, freq) in enumerate(words[:2]):
            if i:
                size = int(round((0.5 * freq + 0.5) * size))
            while size >= 4 and not fits(word, size):
                size -= 1
            sizes.append(size)
        font_size = int(2 * sizes[0] * sizes[1] / (sizes[0] + sizes[1])) if sum(sizes) else canvas_h

    rng = random.Random(123)
    occupied = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    placed = []
    last_freq = 1.0
    for word, freq in words:
        font_size = int(round((0.5 * (freq / last_freq) + 0.5) * font_size))
        vertical = rng.random() >= 0.9
        tried_other = False
        spot = None
        while font_size >= 4:
            w, h, _, _ = _word_ink(word, font_size)
            box = (h, w) if vertical else (w, h)
            spot = _free_spot(occupied, box[0] + margin, box[1] + margin, rng)
            if spot is not None:
                break
            if not tried_other:
                vertical, tried_other = not vertical, True
            else:
                font_size -= 1
                vertical = False
        if font_size < 4 or spot is None:
            break
        y, x = spot[0] + margin // 2, spot[1] + margin // 2
        w, h, baseline, rects = _word_ink(word, font_size)
        for rx0, ry0, rx1, ry1 in rects:
            if vertical:
                # Rotated 90 degrees counter-clockwise: reads bottom to top.
                occupied[y + w - rx1 : y + w - rx0, x + ry0 : x + ry1] = 1
            else:
                occupied[y + ry0 : y + ry1, x + rx0 : x + rx1] = 1
        color = svg.lut_color(VIRIDIS, rng.uniform(0, 1))
        placed.append((word, font_size, x, y, w, h, baseline, vertical, color))
        last_freq = freq

    # imshow of the 600x400 image in the default axes, aspect equal: height-limited and centred.
    ax_l, ax_t, ax_r, ax_b = svg.subplot_rect(width, height)
    img_h = ax_b - ax_t
    img_w = img_h * canvas_w / canvas_h
    ox = (ax_l + ax_r) / 2 - img_w / 2
    k = img_w / canvas_w
    parts = [svg.rect(ox, ax_t, ox + img_w, ax_b, WHITE)]
    for word, size, x, y, w, _h, baseline, vertical, color in placed:
        if vertical:
            px, py = ox + (x + baseline) * k, ax_t + (y + w) * k
            transform = f' transform="rotate(-90 {px:.2f} {py:.2f})"'
        else:
            px, py = ox + x * k, ax_t + (y + baseline) * k
            transform = ""
        parts.append(
            f'<text x="{px:.2f}" y="{py:.2f}" font-family="{MONO_FONT}" font-size="{size * k:.2f}" '
            f'textLength="{w * k:.2f}" lengthAdjust="spacingAndGlyphs" fill="{color}"{transform}>'
            f"{svg.xml_text(word)}</text>"
        )
    return svg.svg_document(width, height, "".join(parts))


# --- interactions (plot.scatter_pairwise) ---------------------------------------------------


def _light_palette(color: str) -> List[str]:
    gray = LIGHT_PALETTE_GRAY.get(color.lower()) or svg.blend("#f0f1f3", color, 0.0)
    return [svg.blend(gray, color, i / 255) for i in range(256)]


def scatter_pairwise(config: Settings, x: pd.Series, y: pd.Series, x_label: str, y_label: str) -> str:
    width, height = 576.0, 396.0
    color = _primary(config)
    xs, ys = x.to_numpy(dtype=float), y.to_numpy(dtype=float)
    valid = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[valid], ys[valid]
    parts: List[str] = []
    if len(x) > config.plot.scatter_threshold:
        hexes, (xmin, xmax, ymin, ymax) = _hexbin(xs, ys, 15)
        xaxis = Axis(_margins(xmin, xmax), True, label=x_label)
        yaxis = Axis(_margins(ymin, ymax), False, label=y_label)
        ax = Axes(xaxis, yaxis)
        ax.set_rect(svg.subplot_rect(width, height))
        lut = _light_palette(color)
        counts = np.array([h[2] for h in hexes]) if hexes else np.array([0.0])
        lo, hi = counts.min(), counts.max()
        for cx, cy, count, poly in hexes:
            t = (count - lo) / (hi - lo) if hi > lo else 0.0
            fill = svg.lut_color(lut, t)
            pts = [(float(ax.x(cx + dx)), float(ax.y(cy + dy))) for dx, dy in poly]
            parts.append(svg.polygon(pts, fill, fill, 0.3))
    else:
        lo_x, hi_x = (xs.min(), xs.max()) if len(xs) else (0.0, 1.0)
        lo_y, hi_y = (ys.min(), ys.max()) if len(ys) else (0.0, 1.0)
        ax = Axes(Axis(_margins(lo_x, hi_x), True, label=x_label), Axis(_margins(lo_y, hi_y), False, label=y_label))
        ax.set_rect(svg.subplot_rect(width, height))
        px, py = ax.x(xs), ax.y(ys)
        parts.append(f'<g fill="{color}" stroke="{color}" stroke-width="0.3">')
        parts += [f'<circle cx="{a:.3f}" cy="{b:.3f}" r="3.5"/>' for a, b in zip(px, py)]
        parts.append("</g>")
    return svg.svg_document(width, height, "".join(parts) + svg.labels_svg(ax.all_labels()))


def _hexbin(x: np.ndarray, y: np.ndarray, gridsize: int):
    """matplotlib.axes.Axes.hexbin (linear scales, counts)."""
    nx = gridsize
    ny = int(nx / math.sqrt(3))
    xmin, xmax = (x.min(), x.max()) if len(x) else (0.0, 1.0)
    ymin, ymax = (y.min(), y.max()) if len(y) else (0.0, 1.0)
    xmin, xmax = svg.nonsingular(xmin, xmax, expander=0.1)
    ymin, ymax = svg.nonsingular(ymin, ymax, expander=0.1)
    padding = 1.0e-9 * (xmax - xmin)
    xmin, xmax = xmin - padding, xmax + padding
    sx, sy = (xmax - xmin) / nx, (ymax - ymin) / ny
    ix, iy = (x - xmin) / sx, (y - ymin) / sy
    ix1, iy1 = np.round(ix).astype(int), np.round(iy).astype(int)
    ix2, iy2 = np.floor(ix).astype(int), np.floor(iy).astype(int)
    nx1, ny1, nx2, ny2 = nx + 1, ny + 1, nx, ny
    d1 = (ix - ix1) ** 2 + 3.0 * (iy - iy1) ** 2
    d2 = (ix - ix2 - 0.5) ** 2 + 3.0 * (iy - iy2 - 0.5) ** 2
    bdist = d1 < d2
    counts1 = np.histogram2d(ix1[bdist], iy1[bdist], bins=(nx1, ny1), range=[(0, nx1), (0, ny1)])[0]
    counts2 = np.histogram2d(ix2[~bdist], iy2[~bdist], bins=(nx2, ny2), range=[(0, nx2), (0, ny2)])[0]
    accum = np.concatenate([counts1.ravel(), counts2.ravel()])
    offsets = np.zeros((len(accum), 2))
    offsets[: nx1 * ny1, 0] = np.repeat(np.arange(nx1), ny1)
    offsets[: nx1 * ny1, 1] = np.tile(np.arange(ny1), nx1)
    offsets[nx1 * ny1 :, 0] = np.repeat(np.arange(nx2) + 0.5, ny2)
    offsets[nx1 * ny1 :, 1] = np.tile(np.arange(ny2), nx2) + 0.5
    offsets[:, 0] = offsets[:, 0] * sx + xmin
    offsets[:, 1] = offsets[:, 1] * sy + ymin
    poly = np.array([[0.5, -0.5], [0.5, 0.5], [0.0, 1.0], [-0.5, 0.5], [-0.5, -0.5], [0.0, -1.0]]) * [sx, sy / 3]
    hexes = [(ox, oy, c, poly) for (ox, oy), c in zip(offsets, accum)]
    return hexes, (xmin, xmax, ymin, ymax)


# --- correlations (plot.correlation_matrix) ---------------------------------------------------


def _correlation_font_size(n: int) -> float:
    for limit, size in ((100, 4), (80, 5), (50, 6), (40, 8)):
        if n > limit:
            return size
    return 10


def correlation_matrix(config: Settings, data: pd.DataFrame, vmin: float = -1) -> str:
    width, height = 576.0, 396.0
    labels = [str(c) for c in data.columns]
    n = len(labels)
    matrix = data.to_numpy(dtype=float)
    left, top, right, bottom = svg.subplot_rect(width, height, bottom=0.2)
    region_w, region_h = right - left, bottom - top
    parent_right = left + region_w * 0.8
    side = min(parent_right - left, region_h)
    x0, y0 = parent_right - side, top + (region_h - side) / 2
    fs = _correlation_font_size(n)
    xaxis = Axis((-0.5, n - 0.5), True, ticks=range(n), labels=labels, fontsize=fs, rotation=90)
    yaxis = Axis((n - 0.5, -0.5), False, ticks=range(n), labels=labels, fontsize=fs)
    ax = Axes(xaxis, yaxis)
    ax.set_rect((x0, y0, x0 + side, y0 + side))
    pixels = np.zeros((n, n, 3), dtype=np.uint8)
    has_nan = False
    for i in range(n):
        for j in range(n):
            v = matrix[i, j]
            if not np.isfinite(v):
                has_nan = True
                fill = "#000000"
            else:
                fill = svg.lut_color(RDBU, (v - vmin) / (1 - vmin))
            pixels[i, j] = [int(fill[k : k + 2], 16) for k in (1, 3, 5)]
    parts = [svg.pixel_image(pixels, x0, y0, x0 + side, y0 + side)] if n else []
    # Colour bar: fraction 0.15, pad 0.05, aspect 20.
    cb_x0 = left + region_w * 0.85
    cb_w = region_h / 20
    cb_y0, cb_y1 = top, bottom
    cb_axis = Axis((vmin, 1.0), False, side="right", tick_marks=True)
    cb = Axes(Axis((0, 1), True, visible=False), cb_axis)
    cb.set_rect((cb_x0, cb_y0, cb_x0 + cb_w, cb_y1))
    parts.append(svg.colorbar_svg(cb_x0, cb_y0, cb_x0 + cb_w, cb_y1, RDBU))
    items = ax.all_labels() + cb.all_labels()
    if has_nan:
        # Legend in the upper right: a black patch for invalid coefficients.
        lw, lh = 82.0, 46.0
        lx, ly = x0 + side - lw - 5, y0 + 5
        parts.append(f'<rect x="{lx:.2f}" y="{ly:.2f}" width="{lw}" height="{lh}" rx="2" fill="#ffffff" '
                     'fill-opacity="0.8" stroke="#cccccc" stroke-width="0.3"/>')
        parts.append(svg.rect(lx + 4, ly + 9, lx + 24, ly + 37, "#000000", WHITE))
        items.append(Label("invalid\ncoefficient", lx + 30, ly + lh / 2, 10, ha="left"))
    return svg.svg_document(width, height, "".join(parts) + cb.tick_marks_svg() + svg.labels_svg(items))


# --- missing values (visualisation/missing.py and the missingno-style plots) -----------------------


def missing_font_size(columns: Sequence[str]) -> float:
    max_len = max(len(c) for c in columns)
    n = len(columns)
    size = 13.0 if n < 20 else 12.0 if n < 40 else 10.0 if n < 60 else 8.0
    return size * min(1.0, 20.0 / max_len)


def missing_bar(config: Settings, notnull_counts: pd.Series, nrows: int, columns: List[str]) -> str:
    width, height = 720.0, 360.0
    fs = missing_font_size(columns)
    color = _primary(config)
    pct = (notnull_counts / nrows).to_numpy(dtype=float)
    counts = [str(int(v)) for v in notnull_counts]
    n = len(columns)
    rect = svg.subplot_rect(width, height, left=0.1, right=0.9, bottom=0.3, top=0.8)
    parts = []
    if n <= 50:
        xaxis = Axis((-0.5, n - 0.5), True, ticks=range(n), labels=columns, fontsize=fs, rotation=45, ha="right")
        top_axis = Axis((-0.5, n - 0.5), True, side="top", ticks=range(n), labels=counts, fontsize=fs,
                        rotation=45, ha="left")
        yaxis = Axis(_bar_ylim(float(np.nanmax(pct)) if n else 1.0), False, fontsize=fs)
        ax = Axes(xaxis, yaxis, [top_axis])
        ax.set_rect(rect)
        for i, p in enumerate(pct):
            parts.append(svg.rect(float(ax.x(i - 0.25)), float(ax.y(p)), float(ax.x(i + 0.25)), float(ax.y(0)), color, WHITE))
    else:
        yaxis = Axis((-0.5, n - 0.5), False, ticks=range(n), labels=columns, fontsize=fs)
        right_axis = Axis((-0.5, n - 0.5), False, side="right", ticks=range(n), labels=counts, fontsize=fs)
        xaxis = Axis(_bar_ylim(float(np.nanmax(pct))), True, fontsize=fs)
        ax = Axes(xaxis, yaxis, [right_axis])
        ax.set_rect(rect)
        for i, p in enumerate(pct):
            parts.append(svg.rect(float(ax.x(0)), float(ax.y(i + 0.25)), float(ax.x(p)), float(ax.y(i - 0.25)), color, WHITE))
    return svg.svg_document(width, height, "".join(parts) + svg.labels_svg(ax.all_labels()))


def missing_matrix(config: Settings, notnull: np.ndarray, columns: List[str], nrows: int) -> str:
    width, height = 720.0, 288.0
    fs = missing_font_size(columns) / 20 * 16
    color = _primary(config)
    n_cols = len(columns)
    show_labels = config.plot.missing.force_labels or n_cols <= 50
    xaxis = Axis((-0.5, n_cols - 0.5), True, side="top", ticks=range(n_cols), labels=columns, fontsize=fs,
                 rotation=45, ha="left", visible=show_labels)
    yaxis = Axis((nrows - 0.5, -0.5), False, ticks=[0, nrows - 1], labels=["1", str(nrows)], fontsize=fs)
    ax = Axes(xaxis, yaxis)
    ax.set_rect(svg.subplot_rect(width, height, left=0.1, right=0.9, bottom=0.2, top=0.7))
    left, top, right, bottom = ax.rect
    rgb = np.array([int(round(c * 255)) for c in svg.hex_to_rgb(color)], dtype=np.uint8)
    pixels = np.full(notnull.shape + (3,), 255, dtype=np.uint8)
    pixels[notnull] = rgb
    parts = [svg.pixel_image(pixels, left, top, right, bottom)]
    cells = "".join(parts)
    lines = "".join(svg.line(float(ax.x(j + 0.5)), top, float(ax.x(j + 0.5)), bottom, WHITE, 1.75) for j in range(n_cols - 1))
    return svg.svg_document(width, height, cells + lines + svg.labels_svg(ax.all_labels()))


def _relative_luminance(color: str) -> float:
    rgb = np.array(svg.hex_to_rgb(color))
    rgb = np.where(rgb <= 0.03928, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return float(rgb.dot([0.2126, 0.7152, 0.0722]))


def _heatmap_label(value: float) -> str:
    t = float(format(value, ".2g"))
    if 0.95 <= t < 1:
        return "<1"
    if -1 < t <= -0.95:
        return ">-1"
    if t == 1:
        return "1"
    if t == -1:
        return "-1"
    if -0.05 < t < 0.05:
        return ""
    return str(round(t, 1))


def _seaborn_tick_step(n: int, length: float) -> int:
    max_ticks = int(length // 10)
    if max_ticks < 1:
        return 0
    return n // max_ticks + 1 or 1


def missing_heatmap(config: Settings, corr: pd.DataFrame, mask: np.ndarray, columns: List[str]) -> str:
    n = len(columns)
    fig_h = 4
    if n > 10:
        fig_h += int((n - 10) / 5)
    width, height = 720.0, float(min(fig_h, 10) * 72)
    fs = missing_font_size(columns)
    if n > 40:
        fs /= 1.4
        region = svg.subplot_rect(width, height, left=0.1, right=0.9, top=0.9, bottom=0.3)
    else:
        region = svg.subplot_rect(width, height, left=0.2, right=0.9, top=0.8, bottom=0.3)
    left, top, right, bottom = region
    region_w = right - left
    ax_rect = (left, top, left + region_w * 0.8, bottom)
    step_x = _seaborn_tick_step(n, ax_rect[2] - ax_rect[0])
    step_y = _seaborn_tick_step(n, bottom - top)
    xt = list(range(0, n, step_x)) if step_x else []
    yt = list(range(0, n, step_y)) if step_y else []
    xaxis = Axis((0, n), True, ticks=[i + 0.5 for i in xt], labels=[columns[i] for i in xt], fontsize=fs,
                 rotation=45, ha="right")
    yaxis = Axis((n, 0), False, ticks=[i + 0.5 for i in yt], labels=[columns[i] for i in yt], fontsize=fs)
    ax = Axes(xaxis, yaxis)
    ax.set_rect(ax_rect)
    values = corr.to_numpy(dtype=float)
    parts = []
    texts: List[Label] = []
    for i in range(n):
        for j in range(n):
            v = values[i, j]
            if mask[i, j] or not np.isfinite(v):
                continue
            fill = svg.lut_color(RDBU, (v + 1) / 2)
            x0, x1 = float(ax.x(j)), float(ax.x(j + 1))
            y0, y1 = float(ax.y(i)), float(ax.y(i + 1))
            parts.append(svg.rect(x0, y0, x1 + 0.05, y1 + 0.05, fill))
            text = _heatmap_label(v)
            if text and config.plot.missing.force_labels:
                tc = TEXT_DARK if _relative_luminance(fill) > 0.408 else WHITE
                texts.append(Label(text, (x0 + x1) / 2, (y0 + y1) / 2, fs - 2, color=tc))
    cb_x0 = left + region_w * 0.85
    cb_w = (bottom - top) / 20
    cb = Axes(Axis((0, 1), True, visible=False), Axis((-1.0, 1.0), False, side="right", tick_marks=True))
    cb.set_rect((cb_x0, top, cb_x0 + cb_w, bottom))
    parts.append(svg.colorbar_svg(cb_x0, top, cb_x0 + cb_w, bottom, RDBU))
    items = ax.all_labels() + cb.all_labels() + texts
    body = f'<g shape-rendering="crispEdges">{"".join(parts)}</g>'
    return svg.svg_document(width, height, body + cb.tick_marks_svg() + svg.labels_svg(items))

