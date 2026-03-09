"""Batch aggregator for run summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

CSV_FIELDS = [
    "run_id",
    "complex_id",
    "seed",
    "search_space_mode",
    "runtime_sec",
    "n_eval_total",
    "n_pockets_total",
    "n_pockets_used",
    "reduction_ratio",
    "best_score_cheap",
    "best_score_expensive",
]


def _first_non_null(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _load_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _to_row(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": summary.get("run_id"),
        "complex_id": _first_non_null(summary.get("complex_id"), summary.get("input_id")),
        "seed": _first_non_null(summary.get("seed"), summary.get("config_resolved_subset", {}).get("seed")),
        "search_space_mode": summary.get("search_space_mode"),
        "runtime_sec": _first_non_null(summary.get("runtime_sec"), summary.get("total_runtime_sec")),
        "n_eval_total": _first_non_null(summary.get("n_eval_total"), summary.get("total_n_eval")),
        "n_pockets_total": _first_non_null(summary.get("n_pockets_total"), summary.get("n_pockets_detected")),
        "n_pockets_used": summary.get("n_pockets_used"),
        "reduction_ratio": summary.get("reduction_ratio"),
        "best_score_cheap": _first_non_null(summary.get("best_score_cheap"), summary.get("best_over_pockets_cheap")),
        "best_score_expensive": _first_non_null(summary.get("best_score_expensive"), summary.get("best_over_pockets_expensive")),
    }


def aggregate_runs(root_dir: str | Path, output_csv: str | Path | None = None) -> Path:
    root = Path(root_dir).expanduser().resolve()
    out_csv = Path(output_csv).expanduser().resolve() if output_csv else root / "comparison.csv"

    summary_paths = sorted(root.rglob("summary.json"))
    rows = [_to_row(_load_summary(path)) for path in summary_paths]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    return out_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate dockingpp summary.json files into comparison.csv")
    parser.add_argument("root_dir", help="Root folder containing run outputs")
    parser.add_argument("--output", dest="output_csv", default=None, help="Output CSV path")
    args = parser.parse_args()

    out_csv = aggregate_runs(args.root_dir, args.output_csv)
    print(str(out_csv))


if __name__ == "__main__":
    main()
