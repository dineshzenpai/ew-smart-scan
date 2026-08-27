"""Evaluation harness and figures of merit for ES scan policies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from baselines import Scheduler
from ew_env import EWEnvironment


@dataclass(frozen=True)
class TraceRecord:
    time: int
    band: int
    alert: bool
    valid: bool
    occupied: bool
    true_detection: bool
    false_alarm: bool
    reward: float
    emitter_ids: tuple[int, ...]
    emission_starts: Mapping[int, int]
    prediction: Optional[float]


@dataclass(frozen=True)
class RunMetrics:
    probability_of_detection: float
    probability_false_alarm: float
    intercept_rate: float
    average_reward: float
    prediction_accuracy: float
    average_intercept_time_error: float
    time_to_first_intercept: Mapping[str, float]
    detected_emitters: int
    total_emitters: int

    def flat(self) -> Dict[str, float]:
        """A flat numeric representation suitable for tables and CSV output."""
        output = {
            "Pd": self.probability_of_detection,
            "Pfa": self.probability_false_alarm,
            "intercept_rate": self.intercept_rate,
            "average_reward": self.average_reward,
            "prediction_accuracy": self.prediction_accuracy,
            "intercept_time_error": self.average_intercept_time_error,
            "detected_emitters": float(self.detected_emitters),
            "total_emitters": float(self.total_emitters),
        }
        output.update({f"ttfi_{kind}": value for kind, value in self.time_to_first_intercept.items()})
        return output


def run_episode(environment: EWEnvironment, scheduler: Scheduler, horizon: int, seed: int = 0) -> List[TraceRecord]:
    """Run one evaluation episode without leaking diagnostic truth to a policy."""
    environment.reset(seed)
    scheduler.reset(seed)
    trace: List[TraceRecord] = []
    for _ in range(horizon):
        band = scheduler.select_band()
        prediction = scheduler.predict_occupancy(band)
        result = environment.step(band)
        # Only these two fields cross the scheduler/environment boundary.
        scheduler.observe(result.observation, result.reward)
        emitter_ids = result.truth.emitters_by_band.get(band, ())
        trace.append(
            TraceRecord(
                time=result.observation.time,
                band=band,
                alert=result.observation.alert,
                valid=result.observation.valid,
                occupied=result.truth.occupied_bands[band],
                true_detection=result.true_detection,
                false_alarm=result.false_alarm,
                reward=result.reward,
                emitter_ids=emitter_ids,
                emission_starts=result.truth.emission_starts,
                prediction=prediction,
            )
        )
    return trace


def compute_metrics(trace: Sequence[TraceRecord], environment: EWEnvironment) -> RunMetrics:
    """Compute requested figures of merit from evaluator-only trace information."""
    if not trace:
        raise ValueError("trace must contain at least one record")
    valid = np.asarray([record.valid for record in trace], dtype=bool)
    occupied = np.asarray([record.occupied for record in trace], dtype=bool)
    detections = np.asarray([record.true_detection for record in trace], dtype=bool)
    alarms = np.asarray([record.false_alarm for record in trace], dtype=bool)
    detection_opportunities = occupied & valid
    false_alarm_opportunities = ~occupied & valid
    pd = float(detections[detection_opportunities].mean()) if detection_opportunities.any() else float("nan")
    pfa = float(alarms[false_alarm_opportunities].mean()) if false_alarm_opportunities.any() else float("nan")
    rewards = np.asarray([record.reward for record in trace], dtype=float)

    prediction_correct: List[bool] = []
    timing_error: List[float] = []
    active_at: Dict[int, int] = {}
    detected_at: Dict[int, int] = {}
    for record in trace:
        for emitter_id, start in record.emission_starts.items():
            active_at.setdefault(emitter_id, min(record.time, start))
        if record.valid and record.prediction is not None:
            prediction_correct.append((record.prediction >= 0.5) == record.occupied)
        if record.true_detection:
            for emitter_id in record.emitter_ids:
                detected_at.setdefault(emitter_id, record.time)
                timing_error.append(float(record.time - record.emission_starts[emitter_id]))

    type_by_id = {emitter.emitter_id: emitter.emitter_type for emitter in environment.emitters}
    type_latencies: Dict[str, List[float]] = {}
    for emitter in environment.emitters:
        first_active = active_at.get(emitter.emitter_id)
        if first_active is None:
            continue
        # Missed emitters are right-censored at the episode boundary, avoiding
        # a deceptively optimistic latency result.
        first_detection = detected_at.get(emitter.emitter_id, len(trace))
        type_latencies.setdefault(type_by_id[emitter.emitter_id], []).append(float(first_detection - first_active))
    ttfi = {kind: float(np.mean(values)) for kind, values in sorted(type_latencies.items())}
    return RunMetrics(
        probability_of_detection=pd,
        probability_false_alarm=pfa,
        intercept_rate=float(detections.sum() / (len(trace) * environment.config.slot_duration)),
        average_reward=float(rewards.mean()),
        prediction_accuracy=float(np.mean(prediction_correct)) if prediction_correct else float("nan"),
        average_intercept_time_error=float(np.mean(timing_error)) if timing_error else float("nan"),
        time_to_first_intercept=ttfi,
        detected_emitters=len(detected_at),
        total_emitters=len(environment.emitters),
    )


def evaluate(
    environment: EWEnvironment,
    scheduler_factory: Callable[[], Scheduler],
    episodes: int = 20,
    horizon: int = 240,
    seed: int = 100,
) -> List[RunMetrics]:
    """Evaluate fresh policy instances over common-random-number episodes."""
    results: List[RunMetrics] = []
    for episode in range(episodes):
        scheduler = scheduler_factory()
        trace = run_episode(environment, scheduler, horizon=horizon, seed=seed + episode)
        results.append(compute_metrics(trace, environment))
    return results


def average_metrics(runs: Iterable[RunMetrics]) -> RunMetrics:
    runs = list(runs)
    if not runs:
        raise ValueError("at least one run is required")
    numeric_keys = [
        "probability_of_detection",
        "probability_false_alarm",
        "intercept_rate",
        "average_reward",
        "prediction_accuracy",
        "average_intercept_time_error",
    ]
    def mean_or_nan(values: Sequence[float]) -> float:
        finite = [value for value in values if not np.isnan(value)]
        return float(np.mean(finite)) if finite else float("nan")

    means = {key: mean_or_nan([getattr(run, key) for run in runs]) for key in numeric_keys}
    kinds = sorted({kind for run in runs for kind in run.time_to_first_intercept})
    ttfi = {
        kind: mean_or_nan([run.time_to_first_intercept.get(kind, np.nan) for run in runs])
        for kind in kinds
    }
    return RunMetrics(
        **means,
        time_to_first_intercept=ttfi,
        detected_emitters=int(round(np.mean([run.detected_emitters for run in runs]))),
        total_emitters=int(round(np.mean([run.total_emitters for run in runs]))),
    )


def comparison_table(results: Mapping[str, RunMetrics]) -> str:
    """Return a compact Markdown summary table for a comparison README/report."""
    header = "| method | Pd | Pfa | intercept rate | avg. reward | timing error |"
    divider = "|---|---:|---:|---:|---:|---:|"
    rows = [header, divider]
    for name, metric in results.items():
        rows.append(
            "| {name} | {Pd:.3f} | {Pfa:.3f} | {rate:.3f} | {reward:.3f} | {error:.3f} |".format(
                name=name,
                Pd=metric.probability_of_detection,
                Pfa=metric.probability_false_alarm,
                rate=metric.intercept_rate,
                reward=metric.average_reward,
                error=metric.average_intercept_time_error,
            )
        )
    return "\n".join(rows)
