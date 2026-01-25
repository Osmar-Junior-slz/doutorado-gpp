"""Search engine interface definitions."""

from __future__ import annotations

from typing import Any, Callable, Protocol

from dockingpp.data.structs import RunResult


class SearchEngine(Protocol):
    """Protocol for docking search engines."""

    def search(
        self,
        receptor: Any,
        peptide: Any,
        pockets: list[Any],
        cfg: Any,
        score_cheap_fn: Callable[..., float],
        score_expensive_fn: Callable[..., float],
        prior_pocket: Any,
        prior_pose: Any,
        logger: Any,
    ) -> RunResult:
        """Run the search and return results."""

        ...
