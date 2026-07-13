"""Statistical equivalence helpers for bundled I/O A/B MD tests.

The normal-MD A/B cases deliberately compare independent trajectory samples,
not matching frame indices.  Block means reduce temporal autocorrelation and
make the gate report an auditable practical-equivalence bound.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class StatisticalEquivalencePolicy:
    """Acceptance policy for one observable across independent MD replicas."""

    burn_in_frames: int
    block_size: int
    minimum_blocks_per_replica: int
    confidence_z: float
    relative_margin: float
    absolute_margin: float
    maximum_std_ratio: float


@dataclass(frozen=True)
class BlockSummary:
    sample_count: int
    block_count: int
    mean: float
    std: float
    sem: float


def block_means(
    samples: Sequence[float], policy: StatisticalEquivalencePolicy
) -> list[float]:
    """Return non-overlapping post-warmup block means for one replica."""

    if policy.block_size < 2:
        raise ValueError("block_size must be at least 2")
    if policy.burn_in_frames < 0:
        raise ValueError("burn_in_frames must not be negative")
    if policy.minimum_blocks_per_replica < 2:
        raise ValueError("minimum_blocks_per_replica must be at least 2")
    if any(not math.isfinite(value) for value in samples):
        raise ValueError("statistical equivalence samples must be finite")

    post_warmup = list(samples[policy.burn_in_frames :])
    block_count = len(post_warmup) // policy.block_size
    if block_count < policy.minimum_blocks_per_replica:
        raise ValueError(
            "not enough post-warmup samples for block statistics: "
            f"samples={len(samples)}, burn_in={policy.burn_in_frames}, "
            f"block_size={policy.block_size}, blocks={block_count}, "
            f"required={policy.minimum_blocks_per_replica}"
        )

    usable = post_warmup[: block_count * policy.block_size]
    return [
        statistics.fmean(
            usable[index * policy.block_size : (index + 1) * policy.block_size]
        )
        for index in range(block_count)
    ]


def summarize(block_values: Sequence[float]) -> BlockSummary:
    """Summarize independent block means using the sample standard deviation."""

    if len(block_values) < 2:
        raise ValueError("at least two block means are required")
    if any(not math.isfinite(value) for value in block_values):
        raise ValueError("block means must be finite")

    mean = statistics.fmean(block_values)
    std = statistics.stdev(block_values)
    return BlockSummary(
        sample_count=len(block_values),
        block_count=len(block_values),
        mean=mean,
        std=std,
        sem=std / math.sqrt(len(block_values)),
    )


def compare_replicas(
    label: str,
    legacy_replicas: Sequence[Sequence[float]],
    bundled_replicas: Sequence[Sequence[float]],
    policy: StatisticalEquivalencePolicy,
) -> dict[str, float | int]:
    """Require practical mean and fluctuation equivalence across replicas.

    Each legacy/bundled pair receives the same seed, while distinct replica
    pairs use independent seeds. The mean decision therefore uses matched
    block differences and a conservative normal-approximation confidence
    bound: ``abs(delta) + z * SEM(delta) <= practical_margin``. The standard
    deviation ratio is a separate guard against a bundled path that preserves
    a mean but changes the ensemble fluctuations.
    """

    if len(legacy_replicas) != len(bundled_replicas):
        raise AssertionError(
            f"{label} replica count mismatch: legacy={len(legacy_replicas)} "
            f"bundled={len(bundled_replicas)}"
        )
    if len(legacy_replicas) == 0:
        raise AssertionError(f"{label} has no replicas")
    if policy.confidence_z <= 0.0:
        raise ValueError("confidence_z must be positive")
    if policy.relative_margin < 0.0 or policy.absolute_margin < 0.0:
        raise ValueError("equivalence margins must not be negative")
    if policy.maximum_std_ratio < 1.0:
        raise ValueError("maximum_std_ratio must be at least one")

    paired_blocks = [
        (block_means(legacy_replica, policy), block_means(bundled_replica, policy))
        for legacy_replica, bundled_replica in zip(legacy_replicas, bundled_replicas)
    ]
    for replica_index, (legacy_blocks, bundled_blocks) in enumerate(paired_blocks):
        if len(legacy_blocks) != len(bundled_blocks):
            raise AssertionError(
                f"{label} block count mismatch in replica {replica_index}: "
                f"legacy={len(legacy_blocks)}, bundled={len(bundled_blocks)}"
            )
    legacy_blocks = [block for blocks, _ in paired_blocks for block in blocks]
    bundled_blocks = [block for _, blocks in paired_blocks for block in blocks]
    legacy = summarize(legacy_blocks)
    bundled = summarize(bundled_blocks)
    paired_deltas = [
        bundled_block - legacy_block
        for legacy_blocks, bundled_blocks in paired_blocks
        for legacy_block, bundled_block in zip(legacy_blocks, bundled_blocks)
    ]
    delta = statistics.fmean(paired_deltas)
    delta_sem = statistics.stdev(paired_deltas) / math.sqrt(len(paired_deltas))
    scale = max(1.0, abs(legacy.mean), abs(bundled.mean), legacy.std, bundled.std)
    margin = max(policy.absolute_margin, policy.relative_margin * scale)
    confidence_bound = abs(delta) + policy.confidence_z * delta_sem
    if confidence_bound > margin:
        raise AssertionError(
            f"{label} mean is not statistically equivalent: "
            f"legacy_mean={legacy.mean:.12g}, bundled_mean={bundled.mean:.12g}, "
            f"delta={delta:.12g}, sem={delta_sem:.12g}, "
            f"confidence_bound={confidence_bound:.12g}, margin={margin:.12g}"
        )

    zero_scale = max(policy.absolute_margin, 1.0e-15)
    if legacy.std <= zero_scale and bundled.std <= zero_scale:
        std_ratio = 1.0
    elif min(legacy.std, bundled.std) <= zero_scale:
        raise AssertionError(
            f"{label} fluctuation mismatch: legacy_std={legacy.std:.12g}, "
            f"bundled_std={bundled.std:.12g}"
        )
    else:
        std_ratio = max(legacy.std, bundled.std) / min(legacy.std, bundled.std)
        if std_ratio > policy.maximum_std_ratio:
            raise AssertionError(
                f"{label} fluctuation mismatch: legacy_std={legacy.std:.12g}, "
                f"bundled_std={bundled.std:.12g}, ratio={std_ratio:.12g}, "
                f"maximum={policy.maximum_std_ratio:.12g}"
            )

    return {
        "legacy_block_count": legacy.block_count,
        "bundled_block_count": bundled.block_count,
        "paired_block_count": len(paired_deltas),
        "legacy_mean": legacy.mean,
        "bundled_mean": bundled.mean,
        "legacy_std": legacy.std,
        "bundled_std": bundled.std,
        "mean_delta": delta,
        "mean_delta_sem": delta_sem,
        "confidence_bound": confidence_bound,
        "practical_margin": margin,
        "std_ratio": std_ratio,
    }
