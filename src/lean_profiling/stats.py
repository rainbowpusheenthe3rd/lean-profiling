"""The handful of scipy routines ydata-profiling relied on, in numpy."""

import math
from typing import Dict

import numpy as np
import pandas as pd


def _gammaincc(a: float, x: float) -> float:
    """Regularised upper incomplete gamma Q(a, x) (Numerical Recipes 6.2)."""
    if x < 0 or a <= 0:
        return math.nan
    if x == 0:
        return 1.0
    log_prefix = -x + a * math.log(x) - math.lgamma(a)
    if x < a + 1:
        # Series for P(a, x), then Q = 1 - P.
        term = total = 1.0 / a
        ap = a
        for _ in range(10000):
            ap += 1
            term *= x / ap
            total += term
            if abs(term) < abs(total) * 1e-16:
                break
        return max(0.0, 1.0 - total * math.exp(log_prefix))
    # Continued fraction for Q(a, x), modified Lentz.
    tiny = 1e-300
    b = x + 1 - a
    c = 1 / tiny
    d = 1 / b
    h = d
    for i in range(1, 10000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-16:
            break
    return math.exp(log_prefix) * h


def chi2_sf(statistic: float, dof: float) -> float:
    """Survival function of the chi-squared distribution (scipy.stats.chi2.sf)."""
    if dof <= 0 or math.isnan(statistic):
        return math.nan
    return _gammaincc(dof / 2.0, statistic / 2.0)


def chisquare(observed: np.ndarray) -> Dict[str, float]:
    """One-way chi-square test against a uniform distribution (scipy.stats.chisquare)."""
    observed = np.asarray(observed, dtype=float)
    expected = observed.mean()
    statistic = float(((observed - expected) ** 2 / expected).sum())
    return {"statistic": statistic, "pvalue": chi2_sf(statistic, observed.size - 1)}


def chi2_contingency_statistic(observed: np.ndarray, correction: bool = True) -> float:
    """Chi-square statistic of a contingency table (scipy.stats.chi2_contingency()[0])."""
    observed = np.asarray(observed, dtype=float)
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.sum()
    dof = expected.size - sum(expected.shape) + expected.ndim - 1
    if dof == 0:
        return 0.0
    if dof == 1 and correction:
        diff = expected - observed
        observed = observed + np.sign(diff) * np.minimum(0.5, np.abs(diff))
    return float(((observed - expected) ** 2 / expected).sum())


def entropy_base2(counts: np.ndarray) -> float:
    """Shannon entropy in bits (scipy.stats.entropy(counts, base=2))."""
    p = np.asarray(counts, dtype=float)
    p = p / p.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def spearman(x: pd.Series, y: pd.Series) -> float:
    """Spearman rank correlation on pairwise-complete observations (pandas Series.corr)."""
    valid = x.notna().to_numpy() & y.notna().to_numpy()
    if valid.sum() < 2:
        return math.nan
    rx = pd.Series(x.to_numpy()[valid]).rank().to_numpy(dtype=float)
    ry = pd.Series(y.to_numpy()[valid]).rank().to_numpy(dtype=float)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = math.sqrt(float((rx * rx).sum() * (ry * ry).sum()))
    if denom == 0:
        return math.nan
    return float(np.clip((rx * ry).sum() / denom, -1.0, 1.0))
