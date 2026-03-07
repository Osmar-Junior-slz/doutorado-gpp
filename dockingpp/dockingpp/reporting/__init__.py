"""Pacote de relatórios do dockingpp."""

from dockingpp.reporting.aggregates import compute_cost_quality, compute_pockets, compute_search_reduction
from dockingpp.reporting.loaders import discover_run_dirs, load_report_bundle
from dockingpp.reporting.models import ArtifactsManifest, MetricEvent, ReportBundle, RunSummary
from dockingpp.reporting.normalize import optional_warnings, to_timeseries

__all__ = [
    "RunSummary",
    "MetricEvent",
    "ArtifactsManifest",
    "ReportBundle",
    "discover_run_dirs",
    "load_report_bundle",
    "optional_warnings",
    "to_timeseries",
    "compute_search_reduction",
    "compute_cost_quality",
    "compute_pockets",
]
