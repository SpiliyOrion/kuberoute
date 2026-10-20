"""Pure, opt-in model selection from bounded recent inference observations."""

import re
from collections import deque
from dataclasses import dataclass
from math import isfinite, lcm
from statistics import median
from typing import Literal, Mapping, Sequence

from kuberoute_gateway.routing import NoEligibleModelError, RoutingDecision

_MAX_MODELS = 8
_MAX_WINDOW_SIZE = 128
_MODEL_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]*\Z")
_RESERVED_IDS = frozenset({"auto", "mock-small", "mock-large"})


@dataclass(frozen=True)
class InferenceObservation:
    """One completed attempt; only successful backend latency is retained."""

    outcome: Literal["success", "error", "timeout"]
    latency_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.outcome not in ("success", "error", "timeout"):
            raise ValueError("Inference outcome is invalid")
        if self.outcome == "success":
            if (
                isinstance(self.latency_seconds, bool)
                or not isinstance(self.latency_seconds, (int, float))
                or not isfinite(self.latency_seconds)
                or self.latency_seconds <= 0
            ):
                raise ValueError("Successful inference latency must be finite and positive")
        elif self.latency_seconds is not None:
            raise ValueError("Failed inference attempts cannot have a successful latency")


@dataclass(frozen=True)
class PerformanceSnapshot:
    """Immutable, bounded recent attempts for one registered model."""

    observations: tuple[InferenceObservation, ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.observations, tuple)
            or len(self.observations) > _MAX_WINDOW_SIZE
            or any(not isinstance(item, InferenceObservation) for item in self.observations)
        ):
            raise ValueError("Performance snapshot is invalid or exceeds the size limit")

    @property
    def completed_attempts(self) -> int:
        return len(self.observations)

    @property
    def successful_attempts(self) -> int:
        return sum(item.outcome == "success" for item in self.observations)

    @property
    def failed_attempts(self) -> int:
        return self.completed_attempts - self.successful_attempts

    @property
    def timed_out_attempts(self) -> int:
        return sum(item.outcome == "timeout" for item in self.observations)

    @property
    def successful_latencies_seconds(self) -> tuple[float, ...]:
        return tuple(
            float(item.latency_seconds)
            for item in self.observations
            if item.outcome == "success" and item.latency_seconds is not None
        )


def _validate_model_ids(model_ids: Sequence[str]) -> tuple[str, ...]:
    ids = tuple(model_ids)
    if len(ids) > _MAX_MODELS:
        raise ValueError("Eligible model IDs must be unique and limited to eight")
    if any(
        not isinstance(model_id, str)
        or not 1 <= len(model_id) <= 128
        or not _MODEL_ID_PATTERN.fullmatch(model_id)
        or model_id in _RESERVED_IDS
        for model_id in ids
    ):
        raise ValueError("Eligible model ID is invalid or reserved")
    if len(ids) != len(set(ids)):
        raise ValueError("Eligible model IDs must be unique and limited to eight")
    return ids


class RollingPerformanceWindow:
    """Hold at most ``window_size`` completed attempts per eligible model."""

    def __init__(self, model_ids: Sequence[str], *, window_size: int = 32) -> None:
        if (
            isinstance(window_size, bool)
            or not isinstance(window_size, int)
            or not 1 <= window_size <= _MAX_WINDOW_SIZE
        ):
            raise ValueError("Observation window size must be between 1 and 128")
        self._observations: dict[str, deque[InferenceObservation]] = {
            model_id: deque(maxlen=window_size) for model_id in _validate_model_ids(model_ids)
        }

    def record(self, model_id: str, observation: InferenceObservation) -> None:
        if not isinstance(observation, InferenceObservation):
            raise ValueError("Inference observation is invalid")
        try:
            self._observations[model_id].append(observation)
        except KeyError:
            raise ValueError("Model is not tracked by the performance window") from None

    def snapshot(self, model_id: str) -> PerformanceSnapshot:
        try:
            return PerformanceSnapshot(tuple(self._observations[model_id]))
        except KeyError:
            raise ValueError("Model is not tracked by the performance window") from None

    def snapshots(self) -> dict[str, PerformanceSnapshot]:
        return {model_id: self.snapshot(model_id) for model_id in self._observations}


@dataclass(frozen=True)
class PerformancePolicyConfig:
    """Fixed policy settings; the decision counter is supplied by a future caller."""

    window_size: int = 32
    min_samples: int = 3
    exploration_interval: int = 4
    ongoing_exploration_interval: int = 16
    failure_probe_interval: int = 64
    failure_rate_threshold: float = 0.5
    failure_penalty: float = 1.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.window_size, bool)
            or not isinstance(self.window_size, int)
            or not 1 <= self.window_size <= _MAX_WINDOW_SIZE
            or isinstance(self.min_samples, bool)
            or not isinstance(self.min_samples, int)
            or not 1 <= self.min_samples <= self.window_size
        ):
            raise ValueError("Performance sample settings are invalid")
        if (
            isinstance(self.exploration_interval, bool)
            or not isinstance(self.exploration_interval, int)
            or not 2 <= self.exploration_interval <= 1000
        ):
            raise ValueError("Exploration interval must be between 2 and 1000")
        if (
            isinstance(self.ongoing_exploration_interval, bool)
            or not isinstance(self.ongoing_exploration_interval, int)
            or not self.exploration_interval < self.ongoing_exploration_interval <= 1000
            or isinstance(self.failure_probe_interval, bool)
            or not isinstance(self.failure_probe_interval, int)
            or not self.ongoing_exploration_interval < self.failure_probe_interval <= 1000
        ):
            raise ValueError("Exploration intervals must increase and be at most 1000")
        if (
            isinstance(self.failure_rate_threshold, bool)
            or not isinstance(self.failure_rate_threshold, (int, float))
            or not isfinite(self.failure_rate_threshold)
            or not 0 < self.failure_rate_threshold < 1
        ):
            raise ValueError("Failure-rate threshold must be finite and between 0 and 1")
        if (
            isinstance(self.failure_penalty, bool)
            or not isinstance(self.failure_penalty, (int, float))
            or not isfinite(self.failure_penalty)
            or not 0 < self.failure_penalty <= 4
        ):
            raise ValueError("Failure penalty must be finite and greater than 0, up to 4")


def select_performance_model(
    eligible_model_ids: Sequence[str],
    snapshots: Mapping[str, PerformanceSnapshot],
    config: PerformancePolicyConfig,
    *,
    decision_index: int,
) -> RoutingDecision:
    """Select a registered eligible model without doing I/O or executing inference.

    Unscored candidates are sampled frequently. Scored losers receive slower
    round-robin probes; failure-heavy candidates receive only rare probes when
    a healthier candidate exists. The caller supplies a monotonic index and feedback.
    """
    ids = _validate_model_ids(eligible_model_ids)
    if not ids:
        raise NoEligibleModelError("No eligible inference model is configured")
    if (
        isinstance(decision_index, bool)
        or not isinstance(decision_index, int)
        or decision_index < 0
    ):
        raise ValueError("Decision index must be a nonnegative integer")
    if len(ids) == 1:
        return RoutingDecision(ids[0], "performance_v1")

    recent = {model_id: snapshots.get(model_id, PerformanceSnapshot()) for model_id in ids}
    if any(
        not isinstance(snapshot, PerformanceSnapshot)
        or snapshot.completed_attempts > config.window_size
        for snapshot in recent.values()
    ):
        raise ValueError("Performance snapshot is invalid or exceeds the configured window")

    scored = [
        model_id
        for model_id in ids
        if recent[model_id].completed_attempts >= config.min_samples
        and recent[model_id].successful_attempts > 0
    ]
    failure_heavy = [
        model_id
        for model_id in ids
        if recent[model_id].completed_attempts >= config.min_samples
        and recent[model_id].failed_attempts / recent[model_id].completed_attempts
        >= config.failure_rate_threshold
    ]
    healthy_scored = [model_id for model_id in scored if model_id not in failure_heavy]
    under_sampled = [
        model_id for model_id in ids if recent[model_id].completed_attempts < config.min_samples
    ]

    def adjusted_latency(model_id: str) -> float:
        snapshot = recent[model_id]
        success_rate = snapshot.successful_attempts / snapshot.completed_attempts
        return median(snapshot.successful_latencies_seconds) / (
            success_rate**config.failure_penalty
        )

    if not healthy_scored:
        if under_sampled:
            selected = min(under_sampled, key=lambda model_id: recent[model_id].completed_attempts)
        elif scored:
            winner = min(scored, key=adjusted_latency)
            alternatives = [model_id for model_id in ids if model_id != winner]
            decision_number = decision_index + 1
            if alternatives and decision_number % config.ongoing_exploration_interval == 0:
                probe_index = decision_number // config.ongoing_exploration_interval - 1
                selected = alternatives[probe_index % len(alternatives)]
            else:
                selected = winner
        else:
            selected = min(ids, key=lambda model_id: recent[model_id].completed_attempts)
    else:
        winner = min(healthy_scored, key=adjusted_latency)
        decision_number = decision_index + 1
        if failure_heavy and decision_number % config.failure_probe_interval == 0:
            probe_index = decision_number // config.failure_probe_interval - 1
            selected = failure_heavy[probe_index % len(failure_heavy)]
        elif under_sampled and decision_number % config.exploration_interval == 0:
            selected = min(under_sampled, key=lambda model_id: recent[model_id].completed_attempts)
        else:
            scored_losers = [model_id for model_id in healthy_scored if model_id != winner]
            if scored_losers and decision_number % config.ongoing_exploration_interval == 0:
                probe_index = decision_number // config.ongoing_exploration_interval - 1
                if failure_heavy:
                    overlap = lcm(
                        config.ongoing_exploration_interval, config.failure_probe_interval
                    )
                    probe_index -= (decision_number - 1) // overlap
                selected = scored_losers[probe_index % len(scored_losers)]
            else:
                selected = winner
    return RoutingDecision(selected, "performance_v1")
