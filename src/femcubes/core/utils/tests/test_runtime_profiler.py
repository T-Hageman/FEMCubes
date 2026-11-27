"""Tests for cumulative runtime profiling."""

import pytest

from femcubes.core.utils.runtime_profiler import (
    RUNTIME_TIMING_CATEGORIES,
    RuntimeProfiler,
)


def test_disabled_profiler_returns_known_zero_categories_without_reading_clock():
    def _unexpected_clock_call():
        raise AssertionError("disabled profiling must not read the clock")

    profiler = RuntimeProfiler(enabled=False, clock=_unexpected_clock_call)

    with profiler.measure("local_assembly"):
        pass

    assert profiler.snapshot() == {
        category: 0.0 for category in RUNTIME_TIMING_CATEGORIES
    }
    assert profiler.snapshot_counts() == {
        category: 0 for category in RUNTIME_TIMING_CATEGORIES
    }


def test_profiler_accumulates_repeated_regions_and_records_exceptional_exit():
    clock_values = iter((1.0, 1.25, 2.0, 2.75, 4.0, 4.5))
    profiler = RuntimeProfiler(enabled=True, clock=lambda: next(clock_values))

    with profiler.measure("local_assembly"):
        pass
    with profiler.measure("local_assembly"):
        pass
    with pytest.raises(RuntimeError, match="failed"):
        with profiler.measure("linear_solve"):
            raise RuntimeError("failed")

    timings = profiler.snapshot()
    counts = profiler.snapshot_counts()
    assert timings["local_assembly"] == pytest.approx(1.0)
    assert timings["linear_solve"] == pytest.approx(0.5)
    assert timings["refinement"] == 0.0
    assert counts["local_assembly"] == 2
    assert counts["linear_solve"] == 1
    assert counts["refinement"] == 0


def test_profiler_reset_clears_accumulated_values():
    clock_values = iter((3.0, 4.0))
    profiler = RuntimeProfiler(enabled=True, clock=lambda: next(clock_values))
    with profiler.measure("output"):
        pass

    profiler.reset()

    assert profiler.snapshot()["output"] == 0.0
    assert profiler.snapshot_counts()["output"] == 0


def test_profiler_records_untimed_custom_events():
    profiler = RuntimeProfiler(enabled=True)

    profiler.count("residual_only_assembly")
    profiler.count("residual_only_assembly")

    assert profiler.snapshot_counts()["residual_only_assembly"] == 2
    assert profiler.snapshot().get("residual_only_assembly", 0.0) == 0.0


def test_profiler_reclassifies_part_of_an_accumulated_region():
    clock_values = iter((2.0, 5.0))
    profiler = RuntimeProfiler(enabled=True, clock=lambda: next(clock_values))
    with profiler.measure("refinement"):
        pass

    profiler.reclassify(1.25, source="refinement", target="plotting")

    timings = profiler.snapshot()
    assert timings["refinement"] == pytest.approx(1.75)
    assert timings["plotting"] == pytest.approx(1.25)


def test_nested_regions_are_recorded_as_exclusive_times():
    clock_values = iter((1.0, 2.0, 5.0, 7.0))
    profiler = RuntimeProfiler(enabled=True, clock=lambda: next(clock_values))

    with profiler.measure("assembly_bookkeeping"):
        with profiler.measure("local_assembly"):
            pass

    timings = profiler.snapshot()
    assert timings["assembly_bookkeeping"] == pytest.approx(3.0)
    assert timings["local_assembly"] == pytest.approx(3.0)
    assert sum(
        timings[category]
        for category in ("assembly_bookkeeping", "local_assembly")
    ) == pytest.approx(6.0)
