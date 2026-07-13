"""Semantic observable series for trajectory A/B comparisons."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence


def trajectory_observable_series(
    dataset: str,
    values: Sequence[float],
    shape: tuple[int, ...],
    *,
    atom_weights: Sequence[float] | None = None,
) -> dict[str, list[float]]:
    """Convert a particle or box trajectory into physical frame observables."""

    if dataset.endswith("/box/edges/value") and len(shape) == 3:
        return _box_observables(values, shape)
    if len(shape) != 3 or shape[-1] != 3:
        return {}
    if atom_weights is not None and (
        len(atom_weights) != shape[1]
        or any(weight <= 0.0 for weight in atom_weights)
    ):
        raise AssertionError(
            "atom weights must be positive and match atom count"
        )
    if dataset.endswith("/position/value"):
        return _position_observables(values, shape, atom_weights)
    if dataset.endswith("/velocity/value"):
        return _velocity_observables(values, shape, atom_weights)
    if dataset.endswith("/force/value"):
        return _force_observables(values, shape)
    return {}


def _vector_frames(
    values: Sequence[float], shape: tuple[int, ...]
) -> list[list[tuple[float, float, float]]]:
    frame_count, item_count, width = shape
    if frame_count <= 0 or item_count <= 0 or width != 3:
        raise AssertionError(f"invalid vector trajectory shape: {shape}")
    if len(values) != frame_count * item_count * width:
        raise AssertionError(
            f"trajectory shape/value mismatch: shape={shape}, values={len(values)}"
        )
    frames = []
    offset = 0
    for _ in range(frame_count):
        frame = []
        for _ in range(item_count):
            frame.append(
                (values[offset], values[offset + 1], values[offset + 2])
            )
            offset += 3
        frames.append(frame)
    return frames


def _quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _norm(vector: tuple[float, float, float]) -> float:
    return math.sqrt(sum(component * component for component in vector))


def _position_observables(
    values: Sequence[float],
    shape: tuple[int, ...],
    atom_weights: Sequence[float] | None,
) -> dict[str, list[float]]:
    frames = _vector_frames(values, shape)
    origin = frames[0]
    weights = (
        list(atom_weights) if atom_weights is not None else [1.0] * shape[1]
    )
    center_name = (
        "center_of_mass" if atom_weights is not None else "center_of_geometry"
    )
    total_weight = sum(weights)
    result = {
        f"{center_name}_component_{axis}": [
            sum(weight * vector[axis] for weight, vector in zip(weights, frame))
            / total_weight
            for frame in frames
        ]
        for axis in range(3)
    }
    for atom_index in range(shape[1]):
        result[f"atom_{atom_index}_squared_displacement"] = [
            sum(
                (frame[atom_index][axis] - origin[atom_index][axis]) ** 2
                for axis in range(3)
            )
            for frame in frames
        ]
    pair_distances = []
    for frame in frames:
        distances = [
            _norm(
                tuple(
                    frame[right][axis] - frame[left][axis] for axis in range(3)
                )
            )
            for left in range(len(frame))
            for right in range(left + 1, len(frame))
        ]
        pair_distances.append(distances or [0.0])
    for name, fraction in (("q25", 0.25), ("median", 0.5), ("q75", 0.75)):
        result[f"pair_distance_{name}"] = [
            _quantile(distances, fraction) for distances in pair_distances
        ]
    return result


def _velocity_observables(
    values: Sequence[float],
    shape: tuple[int, ...],
    atom_weights: Sequence[float] | None,
) -> dict[str, list[float]]:
    frames = _vector_frames(values, shape)
    result: dict[str, list[float]] = {}
    for axis in range(3):
        result[f"component_{axis}_mean"] = [
            statistics.fmean(vector[axis] for vector in frame)
            for frame in frames
        ]
        result[f"component_{axis}_variance"] = [
            statistics.pvariance(vector[axis] for vector in frame)
            for frame in frames
        ]
    if atom_weights is None:
        result["mean_squared_speed"] = [
            statistics.fmean(_norm(vector) ** 2 for vector in frame)
            for frame in frames
        ]
    else:
        total_mass = sum(atom_weights)
        result["mass_weighted_mean_squared_speed"] = [
            sum(
                weight * _norm(vector) ** 2
                for weight, vector in zip(atom_weights, frame)
            )
            / total_mass
            for frame in frames
        ]
    return _add_norm_distribution(result, frames)


def _force_observables(
    values: Sequence[float], shape: tuple[int, ...]
) -> dict[str, list[float]]:
    frames = _vector_frames(values, shape)
    result: dict[str, list[float]] = {
        f"net_component_{axis}": [
            sum(vector[axis] for vector in frame) for frame in frames
        ]
        for axis in range(3)
    }
    result["component_rms"] = [
        math.sqrt(
            statistics.fmean(
                component * component
                for vector in frame
                for component in vector
            )
        )
        for frame in frames
    ]
    for atom_index in range(shape[1]):
        result[f"atom_{atom_index}_norm"] = [
            _norm(frame[atom_index]) for frame in frames
        ]
    return _add_norm_distribution(result, frames)


def _add_norm_distribution(
    result: dict[str, list[float]],
    frames: Sequence[Sequence[tuple[float, float, float]]],
) -> dict[str, list[float]]:
    frame_norms = [[_norm(vector) for vector in frame] for frame in frames]
    for name, fraction in (("q25", 0.25), ("median", 0.5), ("q75", 0.75)):
        result[f"norm_{name}"] = [
            _quantile(norms, fraction) for norms in frame_norms
        ]
    return result


def _box_observables(
    values: Sequence[float], shape: tuple[int, ...]
) -> dict[str, list[float]]:
    if shape[1:] != (3, 3) or len(values) != shape[0] * 9:
        raise AssertionError(f"invalid box trajectory shape: {shape}")
    matrices = [
        values[index * 9 : (index + 1) * 9] for index in range(shape[0])
    ]
    result = {
        f"matrix_{row}_{column}": [
            matrix[row * 3 + column] for matrix in matrices
        ]
        for row in range(3)
        for column in range(3)
    }
    vectors = [
        [
            tuple(matrix[row * 3 + column] for column in range(3))
            for row in range(3)
        ]
        for matrix in matrices
    ]
    for axis in range(3):
        result[f"length_{axis}"] = [_norm(matrix[axis]) for matrix in vectors]
    result["volume"] = [abs(_determinant(matrix)) for matrix in matrices]
    for left, right in ((0, 1), (0, 2), (1, 2)):
        result[f"angle_{left}_{right}"] = [
            _angle(matrix[left], matrix[right]) for matrix in vectors
        ]
    return result


def _determinant(matrix: Sequence[float]) -> float:
    return (
        matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7])
        - matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6])
        + matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6])
    )


def _angle(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> float:
    denominator = _norm(left) * _norm(right)
    if denominator == 0.0:
        raise AssertionError("box vectors must have non-zero length")
    cosine = sum(a * b for a, b in zip(left, right)) / denominator
    return math.acos(max(-1.0, min(1.0, cosine)))
