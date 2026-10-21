"""Data profiling reports for pandas DataFrames, with no dependencies beyond pandas and numpy.

A drop-in replacement for the parts of ydata-profiling most pipelines use::

    from lean_profiling import ProfileReport

    report = ProfileReport(df, title="Training data")
    report.to_file("profiling.json")
    report.to_file("report.html")
"""

from lean_profiling.report import ProfileReport, __version__

__all__ = ["ProfileReport", "__version__"]
