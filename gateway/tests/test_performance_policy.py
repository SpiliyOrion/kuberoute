"""Pure, deterministic policy behavior without inference or network access."""

from typing import cast

import pytest
from kuberoute_gateway.backends import ChatBackend
from kuberoute_gateway.mock_backends import MockSmallBackend
from kuberoute_gateway.performance_policy import (
    InferenceObservation,
    PerformancePolicyConfig,
    PerformanceSnapshot,
    RollingPerformanceWindow,
    select_performance_model,
)
from kuberoute_gateway.registry import ModelRegistry
from kuberoute_gateway.routing import NoEligibleModelError, RoutingDecision, select_model

CONFIG = PerformancePolicyConfig()


def snapshot(*attempts: tuple[str, float | None]) -> PerformanceSnapshot:
    return PerformanceSnapshot(tuple(InferenceObservation(*attempt) for attempt in attempts))


def success(latency: float) -> tuple[str, float]:
    return ("success", latency)


def test_no_eligible_model_and_one_model() -> None:
    with pytest.raises(NoEligibleModelError, match="No eligible inference model"):
        select_performance_model([], {}, CONFIG, decision_index=0)
    assert select_performance_model(["real-a"], {}, CONFIG, decision_index=0) == RoutingDecision(
        "real-a", "performance_v1"
    )


def test_unmeasured_models_are_balanced_and_not_starved() -> None:
    window = RollingPerformanceWindow(["real-a", "real-b", "real-c"])
    decisions: list[str] = []
    for index in range(9):
        decision = select_performance_model(
            ["real-a", "real-b", "real-c"], window.snapshots(), CONFIG, decision_index=index
        )
        decisions.append(decision.model_id)
        window.record(decision.model_id, InferenceObservation("success", 0.2))
    assert decisions[:3] == ["real-a", "real-b", "real-c"]
    assert set(decisions) == {"real-a", "real-b", "real-c"}


def test_periodic_exploration_when_one_candidate_is_well_measured() -> None:
    stats = {"real-a": snapshot(success(0.2), success(0.2), success(0.2))}
    assert (
        select_performance_model(["real-a", "real-b"], stats, CONFIG, decision_index=2).model_id
        == "real-a"
    )
    assert (
        select_performance_model(["real-a", "real-b"], stats, CONFIG, decision_index=3).model_id
        == "real-b"
    )


def test_lower_recent_median_latency_wins() -> None:
    stats = {
        "real-a": snapshot(success(0.8), success(0.7), success(0.9)),
        "real-b": snapshot(success(0.4), success(0.3), success(0.5)),
    }
    assert select_performance_model(
        ["real-a", "real-b"], stats, CONFIG, decision_index=0
    ) == RoutingDecision("real-b", "performance_v1")


def test_failure_rate_penalizes_fast_but_unreliable_model() -> None:
    stats = {
        "flaky": snapshot(success(0.2), ("timeout", None), ("error", None)),
        "reliable": snapshot(success(0.4), success(0.4), success(0.4)),
    }
    assert (
        select_performance_model(["flaky", "reliable"], stats, CONFIG, decision_index=0).model_id
        == "reliable"
    )
    assert stats["flaky"].completed_attempts == 3
    assert stats["flaky"].successful_attempts == 1
    assert stats["flaky"].failed_attempts == 2
    assert stats["flaky"].timed_out_attempts == 1
    assert stats["flaky"].successful_latencies_seconds == (0.2,)


def test_all_failed_candidate_receives_only_rare_probes() -> None:
    stats = {
        "failed": snapshot(("error", None), ("timeout", None), ("error", None)),
        "healthy": snapshot(success(0.5), success(0.5), success(0.5)),
    }
    assert (
        select_performance_model(["failed", "healthy"], stats, CONFIG, decision_index=0).model_id
        == "healthy"
    )
    assert (
        select_performance_model(["failed", "healthy"], stats, CONFIG, decision_index=63).model_id
        == "failed"
    )


def test_scored_losers_are_probed_in_deterministic_registration_order() -> None:
    stats = {
        "winner": snapshot(success(0.2), success(0.2), success(0.2)),
        "slower-a": snapshot(success(0.5), success(0.5), success(0.5)),
        "slower-b": snapshot(success(0.7), success(0.7), success(0.7)),
    }
    ids = ["winner", "slower-a", "slower-b"]
    assert [
        select_performance_model(ids, stats, CONFIG, decision_index=index).model_id
        for index in (14, 15, 31, 47)
    ] == ["winner", "slower-a", "slower-b", "slower-a"]
    assert select_performance_model(ids, stats, CONFIG, decision_index=31).model_id == "slower-b"


def test_failure_probe_overlap_does_not_starve_a_scored_loser() -> None:
    ids = ["winner", "slower-a", "slower-b", "slower-c", "slower-d", "failed"]
    stats = {
        model_id: snapshot(success(0.2), success(0.2), success(0.2))
        if model_id == "winner"
        else snapshot(success(0.8), success(0.8), success(0.8))
        for model_id in ids[:-1]
    }
    stats["failed"] = snapshot(("timeout", None), ("timeout", None), ("timeout", None))

    assert [
        select_performance_model(ids, stats, CONFIG, decision_index=index).model_id
        for index in (15, 31, 47, 63, 79)
    ] == ["slower-a", "slower-b", "slower-c", "failed", "slower-d"]


def test_all_failure_heavy_candidates_still_select_deterministically() -> None:
    ids = ["failed-a", "failed-b"]
    stats = {
        "failed-a": snapshot(("timeout", None), ("error", None), ("timeout", None)),
        "failed-b": snapshot(("error", None), ("timeout", None), ("error", None)),
    }
    assert select_performance_model(ids, stats, CONFIG, decision_index=0) == RoutingDecision(
        "failed-a", "performance_v1"
    )
    assert select_performance_model(ids, stats, CONFIG, decision_index=63) == RoutingDecision(
        "failed-a", "performance_v1"
    )

    stats["failed-a"] = snapshot(success(0.5), ("timeout", None), ("error", None))
    assert select_performance_model(ids, stats, CONFIG, decision_index=0) == RoutingDecision(
        "failed-a", "performance_v1"
    )
    assert select_performance_model(ids, stats, CONFIG, decision_index=15) == RoutingDecision(
        "failed-b", "performance_v1"
    )


def test_improved_scored_loser_can_become_the_preferred_model() -> None:
    ids = ["steady", "improving"]
    window = RollingPerformanceWindow(ids)
    for _ in range(3):
        window.record("steady", InferenceObservation("success", 0.3))
        window.record("improving", InferenceObservation("success", 0.8))

    chosen: list[str] = []
    for index in range(65):
        selected = select_performance_model(
            ids, window.snapshots(), CONFIG, decision_index=index
        ).model_id
        chosen.append(selected)
        latency = 0.1 if selected == "improving" else 0.3
        window.record(selected, InferenceObservation("success", latency))

    assert [chosen[index] for index in (15, 31, 47, 63)] == ["improving"] * 4
    assert chosen[64] == "improving"


def test_persistently_failing_model_has_bounded_exploration_traffic() -> None:
    ids = ["healthy", "failed"]
    window = RollingPerformanceWindow(ids)
    for _ in range(3):
        window.record("healthy", InferenceObservation("success", 0.4))

    failed_decisions: list[int] = []
    for index in range(256):
        selected = select_performance_model(
            ids, window.snapshots(), CONFIG, decision_index=index
        ).model_id
        if selected == "failed":
            failed_decisions.append(index)
            window.record(selected, InferenceObservation("timeout"))
        else:
            window.record(selected, InferenceObservation("success", 0.4))

    assert failed_decisions == [3, 7, 11, 63, 127, 191, 255]


def test_score_ties_use_eligible_registration_order() -> None:
    stats = {
        "real-a": snapshot(success(0.5), success(0.5), success(0.5)),
        "real-b": snapshot(success(0.5), success(0.5), success(0.5)),
    }
    assert (
        select_performance_model(["real-b", "real-a"], stats, CONFIG, decision_index=0).model_id
        == "real-b"
    )


def test_window_evicts_old_attempts_and_updates_failure_rate() -> None:
    window = RollingPerformanceWindow(["real-a"], window_size=3)
    window.record("real-a", InferenceObservation("timeout"))
    for latency in (0.3, 0.2, 0.1):
        window.record("real-a", InferenceObservation("success", latency))
    stats = window.snapshot("real-a")
    assert stats.completed_attempts == 3
    assert stats.successful_attempts == 3
    assert stats.failed_attempts == 0
    assert stats.timed_out_attempts == 0
    assert stats.successful_latencies_seconds == (0.3, 0.2, 0.1)


def test_mock_models_are_not_eligible_and_production_auto_is_unchanged() -> None:
    registry = ModelRegistry()
    registry.register("mock-small", MockSmallBackend())
    registry.register("custom-named-mock", MockSmallBackend())
    registry.register("real-a", cast(ChatBackend, object()), auto_eligible=True)
    registry.register("real-b", cast(ChatBackend, object()), auto_eligible=True)
    assert registry.auto_eligible_model_ids() == ("real-a", "real-b")
    assert select_model("auto", registry) == RoutingDecision("real-a", "first_eligible")
    assert select_model("mock-small", registry) == RoutingDecision("mock-small", "explicit")
    assert (
        select_performance_model(
            registry.auto_eligible_model_ids(), {}, CONFIG, decision_index=0
        ).model_id
        == "real-a"
    )
    with pytest.raises(ValueError, match="invalid or reserved"):
        select_performance_model(["mock-small", "real-a"], {}, CONFIG, decision_index=0)


@pytest.mark.parametrize(
    "observation",
    [
        ("success", None),
        ("success", float("nan")),
        ("success", float("inf")),
        ("success", -1.0),
        ("success", 0.0),
        ("timeout", 0.2),
        ("unknown", None),
    ],
)
def test_invalid_observations_are_rejected(observation: tuple[str, float | None]) -> None:
    with pytest.raises(ValueError):
        InferenceObservation(*observation)


def test_config_and_snapshot_bounds_are_enforced() -> None:
    with pytest.raises(ValueError):
        PerformancePolicyConfig(window_size=129)
    with pytest.raises(ValueError):
        PerformancePolicyConfig(min_samples=33)
    with pytest.raises(ValueError):
        PerformancePolicyConfig(exploration_interval=1)
    with pytest.raises(ValueError):
        PerformancePolicyConfig(ongoing_exploration_interval=4)
    with pytest.raises(ValueError):
        PerformancePolicyConfig(failure_probe_interval=16)
    with pytest.raises(ValueError):
        PerformancePolicyConfig(failure_rate_threshold=1)
    with pytest.raises(ValueError):
        PerformancePolicyConfig(failure_penalty=0)
    with pytest.raises(ValueError):
        PerformanceSnapshot((InferenceObservation("error"),) * 129)
    with pytest.raises(ValueError):
        RollingPerformanceWindow([f"real-{index}" for index in range(9)])
    with pytest.raises(ValueError):
        select_performance_model(["real-a", "real-a"], {}, CONFIG, decision_index=0)
    with pytest.raises(ValueError):
        select_performance_model(["real-a", "real-b"], {}, CONFIG, decision_index=-1)
    with pytest.raises(ValueError, match="invalid or reserved"):
        select_performance_model([cast(str, [])], {}, CONFIG, decision_index=0)


def test_snapshot_ignores_unrelated_ids_and_never_contains_user_content() -> None:
    unrelated_id = "private prompt and response marker"
    stats = {
        "real-a": snapshot(success(0.4), success(0.4), success(0.4)),
        unrelated_id: snapshot(success(0.01), success(0.01), success(0.01)),
    }
    decision = select_performance_model(["real-a"], stats, CONFIG, decision_index=0)
    assert decision == RoutingDecision("real-a", "performance_v1")
    assert unrelated_id not in repr(decision)
    assert unrelated_id not in repr(stats["real-a"])
    window = RollingPerformanceWindow(["real-a"])
    with pytest.raises(ValueError, match="not tracked") as error:
        window.record(unrelated_id, InferenceObservation("success", 0.1))
    assert unrelated_id not in str(error.value)
