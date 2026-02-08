"""Módulo de relatórios para análise e visualização (PT-BR)."""

from dockingpp.reporting.loaders import extract_series, find_matching_jsonl, load_any_json, load_jsonl
from dockingpp.reporting.plots import (
    plot_cost_quality,
    plot_filter_distribution,
    plot_omega_reduction,
    plot_pocket_rank_effect,
    plot_score_stability,
)

__all__ = [
    "extract_series",
    "find_matching_jsonl",
    "load_any_json",
    "load_jsonl",
    "plot_cost_quality",
    "plot_filter_distribution",
    "plot_omega_reduction",
    "plot_pocket_rank_effect",
    "plot_score_stability",
]
