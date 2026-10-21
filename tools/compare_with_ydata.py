"""Check lean-profiling against ydata-profiling on your own data.

Run ydata-profiling once, in a separate throwaway environment (it is not a dependency):

    pip install "ydata-profiling==4.18.4" "setuptools<81"
    python -c "import pandas as pd; from ydata_profiling import ProfileReport; \
ProfileReport(pd.read_csv('training.csv'), progress_bar=False).to_file('ydata.json')"

Then, in the lean-profiling environment:

    python tools/compare_with_ydata.py training.csv ydata.json

It prints every field of table, variables, correlations, alerts, duplicates and sample
that differs. Charts are left out because they are SVG markup.
"""

import json
import math
import sys

import pandas as pd

from lean_profiling import ProfileReport

SECTIONS = ["table", "variables", "correlations", "alerts", "duplicates", "sample"]


def compare(a, b, path, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for key in dict.fromkeys([*a, *b]):
            if key not in a:
                out.append(f"{path}.{key}: only in ydata-profiling")
            elif key not in b:
                out.append(f"{path}.{key}: only in lean-profiling")
            else:
                compare(a[key], b[key], f"{path}.{key}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: {len(a)} items vs {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, f"{path}[{i}]", out)
    elif isinstance(a, float) and isinstance(b, (int, float)):
        same = a == b or (math.isnan(a) and math.isnan(b)) or abs(a - b) <= 1e-9 * max(1.0, abs(b))
        if not same:
            out.append(f"{path}: {a!r} vs {b!r}")
    elif a != b:
        out.append(f"{path}: {str(a)[:80]!r} vs {str(b)[:80]!r}")


def main() -> None:
    csv_path, ydata_json = sys.argv[1], sys.argv[2]
    mine = json.loads(ProfileReport(pd.read_csv(csv_path)).to_json())
    with open(ydata_json, encoding="utf-8") as f:
        theirs = json.load(f)
    differences: list = []
    for section in SECTIONS:
        compare(mine[section], theirs[section], section, differences)
    print(f"{len(differences)} differences")
    print("\n".join(differences))
    sys.exit(1 if differences else 0)


if __name__ == "__main__":
    main()
