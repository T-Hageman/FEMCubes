"""Focused trigger contracts for post-refinement load balancing."""

from __future__ import annotations

from itertools import permutations
from types import SimpleNamespace

import numpy as np
import pytest

from femcubes.core.mesh.refinement import RefinementConfig
from femcubes.core.mesh.refinement.partition import _match_partition_ranks, _should_repartition


def test_repartition_trigger_is_opt_in_parallel_and_strictly_above_threshold():
    artifacts = SimpleNamespace(
        report=SimpleNamespace(
            occurred=True,
            imbalance_ratio_after=1.5,
        )
    )

    assert not _should_repartition(artifacts, RefinementConfig(), comm_size=2)
    assert not _should_repartition(
        artifacts,
        RefinementConfig(repartition=True, repartition_imbalance_threshold=1.5),
        comm_size=1,
    )
    assert not _should_repartition(
        artifacts,
        RefinementConfig(repartition=True, repartition_imbalance_threshold=1.5),
        comm_size=2,
    )

    artifacts.report.imbalance_ratio_after = np.nextafter(1.5, np.inf)
    assert _should_repartition(
        artifacts,
        RefinementConfig(repartition=True, repartition_imbalance_threshold=1.5),
        comm_size=2,
    )


@pytest.mark.parametrize("overlap", [
    [[0, 5, 0], [0, 0, 4], [6, 0, 0]],
    [[9, 8, 0], [8, 0, 0], [0, 7, 6]],
    [[4, 4, 0], [4, 4, 0], [0, 0, 8]],
])
def test_partition_labels_maximize_retained_cells(overlap):
    overlap = np.asarray(overlap, dtype=np.int64)
    rank_for_part = _match_partition_ranks(overlap)
    assert sorted(rank_for_part) == list(range(3))
    retained = sum(overlap[rank_for_part[part], part] for part in range(3))
    optimum = max(sum(overlap[rank, parts[rank]] for rank in range(3)) for parts in permutations(range(3)))
    assert retained == optimum
