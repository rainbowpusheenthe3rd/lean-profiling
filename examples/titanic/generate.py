"""Profile the Titanic passenger list (Kaggle's training set, 891 rows).

Run from the repository root:

    python examples/titanic/generate.py

Writes report.html and profiling.json next to this script.
"""

from pathlib import Path

import pandas as pd

from lean_profiling import ProfileReport

here = Path(__file__).parent
report = ProfileReport(pd.read_csv(here / "titanic.csv"), title="Titanic Dataset")
report.to_file(here / "profiling.json")
report.to_file(here / "report.html")
