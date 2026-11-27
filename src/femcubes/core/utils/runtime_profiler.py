"""Low-overhead cumulative runtime profiling for solver phases."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
import math
from time import perf_counter


RUNTIME_TIMING_CATEGORIES = (
    "form_evaluation",
    "form_cache_lookup",
    "form_compilation",
    "local_assembly",
    "assembly_communication",
    "boundary_conditions",
    "model_assembly",
    "assembly_bookkeeping",
    "linear_solve",
    "refinement",
    "plotting",
    "output",
)


class RuntimeProfiler:
    """Accumulate exclusive local wall times for named runtime phases.

    Timing is deliberately rank-local. MPI reduction is deferred until the
    end of a run so profiling does not introduce collectives inside nonlinear
    iterations.
    """

    def __init__(
        self,
        enabled: bool = False,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        """Create a profiler controlled by an explicit boolean flag."""

        if not isinstance(enabled, bool):
            raise TypeError("RuntimeProfiler enabled must be a bool.")
        self.enabled = enabled
        self._clock = clock
        self._seconds: defaultdict[str, float] = defaultdict(float)
        self._counts: defaultdict[str, int] = defaultdict(int)
        self._active_regions: list[dict[str, float | str]] = []

    @contextmanager
    def measure(self, category: str) -> Iterator[None]:
        """Accumulate one timed region under ``category``."""

        self._validate_category(category)
        if not self.enabled:
            yield
            return

        self.count(category)
        start = self._clock()
        region: dict[str, float | str] = {
            "category": category,
            "start": start,
            "child_seconds": 0.0,
        }
        self._active_regions.append(region)
        try:
            yield
        finally:
            elapsed = max(0.0, self._clock() - start)
            active_region = self._active_regions.pop()
            if active_region is not region:
                raise RuntimeError("Runtime profiling regions exited out of order.")
            child_seconds = float(region["child_seconds"])
            self._seconds[category] += max(0.0, elapsed - child_seconds)
            if self._active_regions:
                parent = self._active_regions[-1]
                parent["child_seconds"] = (
                    float(parent["child_seconds"]) + elapsed
                )

    def count(self, category: str) -> None:
        """Record one named event without adding a timed region."""

        self._validate_category(category)
        if self.enabled:
            self._counts[category] += 1

    def reclassify(
        self,
        seconds: float,
        *,
        source: str,
        target: str,
    ) -> None:
        """Move elapsed time from one accumulated category to another."""

        self._validate_category(source)
        self._validate_category(target)
        if not isinstance(seconds, (int, float)) or not math.isfinite(seconds):
            raise ValueError("Reclassified runtime must be finite.")
        if seconds < 0.0:
            raise ValueError("Reclassified runtime must be nonnegative.")
        if not self.enabled:
            return

        moved = min(float(seconds), self._seconds[source])
        self._seconds[source] -= moved
        self._seconds[target] += moved

    def snapshot(self) -> dict[str, float]:
        """Return a detached timing snapshot, including known zero categories."""

        categories = dict.fromkeys(RUNTIME_TIMING_CATEGORIES)
        categories.update(self._seconds)
        return {
            category: float(self._seconds.get(category, 0.0))
            for category in categories
        }

    def snapshot_counts(self) -> dict[str, int]:
        """Return detached invocation counts, including known zero categories."""

        categories = dict.fromkeys(RUNTIME_TIMING_CATEGORIES)
        categories.update(self._counts)
        return {
            category: int(self._counts.get(category, 0))
            for category in categories
        }

    def reset(self) -> None:
        """Clear all accumulated timings."""

        self._seconds.clear()
        self._counts.clear()

    @staticmethod
    def _validate_category(category: str) -> None:
        """PRIVATE: Reject invalid timing and event category names."""

        if not isinstance(category, str) or not category:
            raise ValueError("Runtime timing category must be a non-empty string.")
