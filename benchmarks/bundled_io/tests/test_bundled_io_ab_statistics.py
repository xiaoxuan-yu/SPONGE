from __future__ import annotations

import pytest

from benchmarks.bundled_io.ab_statistics import (
    StatisticalEquivalencePolicy,
    block_means,
    compare_replicas,
)

POLICY = StatisticalEquivalencePolicy(
    burn_in_frames=2,
    block_size=2,
    minimum_blocks_per_replica=3,
    confidence_z=3.0,
    relative_margin=0.05,
    absolute_margin=0.01,
    maximum_std_ratio=1.5,
)


def test_block_means_discards_warmup_and_uses_non_overlapping_blocks():
    assert block_means([99.0, 99.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0], POLICY) == [
        2.0,
        6.0,
        10.0,
    ]


def test_statistical_equivalence_accepts_matching_replica_ensembles():
    legacy = [
        [0.0, 0.0, 9.9, 10.1, 10.0, 10.2, 9.8, 10.0],
        [0.0, 0.0, 10.2, 9.8, 10.1, 9.9, 10.0, 10.0],
    ]
    bundled = [
        [0.0, 0.0, 10.0, 10.2, 9.9, 10.1, 9.8, 10.0],
        [0.0, 0.0, 10.1, 9.9, 10.0, 10.0, 10.1, 9.9],
    ]

    result = compare_replicas("temperature", legacy, bundled, POLICY)

    assert result["legacy_block_count"] == 6
    assert result["bundled_block_count"] == 6
    assert result["confidence_bound"] <= result["practical_margin"]


def test_statistical_equivalence_rejects_mean_shift_outside_margin():
    legacy = [[0.0, 0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]]
    bundled = [[0.0, 0.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0]]

    with pytest.raises(AssertionError, match="mean is not statistically equivalent"):
        compare_replicas("temperature", legacy, bundled, POLICY)


def test_statistical_equivalence_rejects_changed_fluctuations():
    legacy = [[0.0, 0.0, 9.8, 9.8, 10.2, 10.2, 10.0, 10.0]]
    bundled = [[0.0, 0.0, 9.9, 9.9, 10.1, 10.1, 10.0, 10.0]]

    with pytest.raises(AssertionError, match="fluctuation mismatch"):
        compare_replicas("temperature", legacy, bundled, POLICY)
