"""Online periodicity estimation and a rendezvous-on-a-cycle scheduler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from baselines import Scheduler
from ew_env import Observation


@dataclass(frozen=True)
class PeriodEstimate:
    period: int
    dwell: int
    phase: int
    confidence: float


class PeriodicityEstimator:
    """Estimate a band's period, on-window, and phase from sparse alerts.

    The estimator scores candidate periods by how strongly observations cluster
    into phase bins.  It works with irregular sampling because only observed
    timestamps are used; unvisited slots are never treated as negative labels.
    """

    def __init__(self, min_period: int = 2, max_period: int = 64, min_observations: int = 12) -> None:
        if min_period < 1 or max_period < min_period:
            raise ValueError("invalid period search range")
        self.min_period, self.max_period = int(min_period), int(max_period)
        self.min_observations = int(min_observations)
        self.reset()

    def reset(self) -> None:
        self.observations: List[Tuple[int, bool]] = []

    def observe(self, time: int, alert: bool) -> None:
        self.observations.append((int(time), bool(alert)))

    @staticmethod
    def _best_window(rate: np.ndarray, support: np.ndarray) -> Tuple[int, int, float]:
        period = len(rate)
        # Search contiguous circular windows.  A detection window normally has
        # modest width; allowing up to half a cycle avoids pathological plans.
        best_start, best_width, best_score = 0, 1, -np.inf
        overall = float(np.average(rate, weights=np.maximum(support, 1)))
        for width in range(1, max(2, period // 2 + 1)):
            for start in range(period):
                indices = (start + np.arange(width)) % period
                outside = np.setdiff1d(np.arange(period), indices, assume_unique=False)
                inside_support = support[indices].sum()
                outside_support = support[outside].sum()
                if inside_support < 2 or outside_support < 2:
                    continue
                inside_rate = float(np.average(rate[indices], weights=support[indices]))
                outside_rate = float(np.average(rate[outside], weights=support[outside]))
                score = (inside_rate - outside_rate) * np.sqrt(inside_support / (inside_support + outside_support))
                if score > best_score:
                    best_start, best_width, best_score = start, width, score
        if not np.isfinite(best_score):
            return 0, 1, 0.0
        return best_start, best_width, max(0.0, best_score - 0.1 * overall)

    def estimate(self) -> Optional[PeriodEstimate]:
        if len(self.observations) < self.min_observations or sum(alert for _, alert in self.observations) < 3:
            return None
        times = np.asarray([time for time, _ in self.observations], dtype=int)
        labels = np.asarray([alert for _, alert in self.observations], dtype=float)
        best: Optional[PeriodEstimate] = None
        for period in range(self.min_period, self.max_period + 1):
            phases = times % period
            support = np.bincount(phases, minlength=period).astype(float)
            positives = np.bincount(phases, weights=labels, minlength=period)
            # Beta(1, 1) smoothing prevents one rare positive from dominating.
            rate = (positives + 1.0) / (support + 2.0)
            phase, dwell, contrast = self._best_window(rate, support)
            coverage = min(1.0, support.size and np.count_nonzero(support) / max(1, period))
            confidence = contrast * coverage * min(1.0, len(times) / (2.0 * period))
            candidate = PeriodEstimate(period, dwell, phase, float(confidence))
            if best is None or candidate.confidence > best.confidence + 1e-9 or (
                abs(candidate.confidence - best.confidence) < 1e-9 and candidate.period < best.period
            ):
                best = candidate
        return best

    def probability_at(self, time: int) -> Optional[float]:
        estimate = self.estimate()
        if estimate is None:
            return None
        phase = time % estimate.period
        distance = (phase - estimate.phase) % estimate.period
        return 0.85 if distance < estimate.dwell else 0.05


class RendezvousScheduler(Scheduler):
    """Meeting strategy: attend the earliest high-confidence periodic window.

    Unknown bands are explored round-robin.  Once a periodic model is learned,
    the next scheduled dwell is selected by earliest arrival, which bounds the
    wait by ``period - dwell + retune_slots`` for an isolated known cycle.
    """

    name = "periodic_rendezvous"

    def __init__(
        self,
        num_bands: int,
        min_period: int = 2,
        max_period: int = 64,
        confidence_threshold: float = 0.12,
        seed: int = 0,
    ) -> None:
        super().__init__(num_bands, seed)
        self.min_period, self.max_period = min_period, max_period
        self.confidence_threshold = float(confidence_threshold)
        self.reset()

    def reset(self, seed: Optional[int] = None) -> None:
        super().reset(seed)
        self.time = 0
        self.age = np.zeros(self.num_bands, dtype=int)
        self._round_robin = 0
        self.estimators = [PeriodicityEstimator(self.min_period, self.max_period) for _ in range(self.num_bands)]

    @staticmethod
    def _wait_until_window(time: int, estimate: PeriodEstimate) -> int:
        phase = time % estimate.period
        offset = (estimate.phase - phase) % estimate.period
        return int(offset)

    def select_band(self) -> int:
        candidates: list[tuple[float, int]] = []
        for band, estimator in enumerate(self.estimators):
            estimate = estimator.estimate()
            if estimate is not None and estimate.confidence >= self.confidence_threshold:
                wait = self._wait_until_window(self.time, estimate)
                # Prefer the most imminent meeting, then confidence.
                candidates.append((wait - 0.10 * estimate.confidence, band))
        if candidates:
            return min(candidates)[1]
        # Deliberate coverage while there is insufficient evidence to model a band.
        oldest = np.flatnonzero(self.age == self.age.max())
        if len(oldest) == 1:
            return int(oldest[0])
        band = self._round_robin
        self._round_robin = (self._round_robin + 1) % self.num_bands
        return int(band)

    def observe(self, observation: Observation, reward: float) -> None:
        self.age += 1
        if observation.valid:
            self.age[observation.band] = 0
            self.estimators[observation.band].observe(observation.time, observation.alert)
        self.time = observation.time + 1

    def predict_occupancy(self, band: int) -> Optional[float]:
        return self.estimators[band].probability_at(self.time)

    def worst_case_wait_bound(self, band: int, retune_slots: int = 1) -> Optional[int]:
        """Return the isolated-cycle interception bound from the learned model."""
        estimate = self.estimators[band].estimate()
        if estimate is None or estimate.confidence < self.confidence_threshold:
            return None
        return max(0, estimate.period - estimate.dwell + int(retune_slots))
