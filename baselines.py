"""Open-loop and lightweight learning schedulers for the ES simulator."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence

import numpy as np

from ew_env import Observation


class Scheduler(ABC):
    """A policy interface intentionally limited to receiver observations."""

    name = "scheduler"

    def __init__(self, num_bands: int, seed: int = 0) -> None:
        if num_bands < 1:
            raise ValueError("num_bands must be positive")
        self.num_bands = int(num_bands)
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(self.seed if seed is None else seed)

    @abstractmethod
    def select_band(self) -> int:
        """Choose the next receive band without access to ground truth."""

    def observe(self, observation: Observation, reward: float) -> None:
        """Consume sensor feedback from the last dwell."""

    def predict_occupancy(self, band: int) -> Optional[float]:
        """Optional estimated occupancy probability used by metric logging."""
        return None


class RoundRobinScheduler(Scheduler):
    name = "round_robin"

    def reset(self, seed: Optional[int] = None) -> None:
        super().reset(seed)
        self._next_band = 0

    def select_band(self) -> int:
        band = self._next_band
        self._next_band = (self._next_band + 1) % self.num_bands
        return band


class RandomSweepScheduler(Scheduler):
    name = "random_sweep"

    def select_band(self) -> int:
        return int(self.rng.integers(self.num_bands))


class StaticPriorityScheduler(Scheduler):
    name = "static_priority"

    def __init__(self, num_bands: int, priorities: Sequence[float], seed: int = 0) -> None:
        super().__init__(num_bands, seed)
        if len(priorities) != num_bands:
            raise ValueError("one priority is required per band")
        weights = np.asarray(priorities, dtype=float)
        if (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("priorities must be non-negative with a positive sum")
        self._probabilities = weights / weights.sum()

    def select_band(self) -> int:
        return int(self.rng.choice(self.num_bands, p=self._probabilities))


class DiscountedUCBScheduler(Scheduler):
    """Discounted UCB for non-stationary (restless) per-band rewards."""

    name = "discounted_ucb"

    def __init__(self, num_bands: int, discount: float = 0.985, exploration: float = 1.4, seed: int = 0) -> None:
        super().__init__(num_bands, seed)
        if not 0 < discount <= 1:
            raise ValueError("discount must be in (0, 1]")
        self.discount, self.exploration = float(discount), float(exploration)
        self.reset()

    def reset(self, seed: Optional[int] = None) -> None:
        super().reset(seed)
        self.counts = np.zeros(self.num_bands, dtype=float)
        self.values = np.zeros(self.num_bands, dtype=float)
        self.total = 0.0

    def select_band(self) -> int:
        unseen = np.flatnonzero(self.counts < 1e-8)
        if len(unseen):
            return int(unseen[0])
        bonus = self.exploration * np.sqrt(np.log(self.total + 1.0) / self.counts)
        return int(np.argmax(self.values / self.counts + bonus))

    def observe(self, observation: Observation, reward: float) -> None:
        if not observation.valid:
            return
        self.counts *= self.discount
        self.values *= self.discount
        self.counts[observation.band] += 1.0
        # Alert is the portable bandit reward; shaped environment reward is
        # also incorporated at a lower weight to discourage false alarms.
        signal = float(observation.alert) + 0.10 * float(reward)
        self.values[observation.band] += signal
        self.total = self.discount * self.total + 1.0

    def predict_occupancy(self, band: int) -> Optional[float]:
        if self.counts[band] < 1e-8:
            return None
        return float(np.clip(self.values[band] / self.counts[band], 0.0, 1.0))


class ThompsonSamplingScheduler(Scheduler):
    """Beta-Bernoulli Thompson sampler with forgetting for restless bands."""

    name = "thompson_sampling"

    def __init__(self, num_bands: int, discount: float = 0.99, seed: int = 0) -> None:
        super().__init__(num_bands, seed)
        self.discount = float(discount)
        self.reset()

    def reset(self, seed: Optional[int] = None) -> None:
        super().reset(seed)
        self.alpha = np.ones(self.num_bands, dtype=float)
        self.beta = np.ones(self.num_bands, dtype=float)

    def select_band(self) -> int:
        return int(np.argmax(self.rng.beta(self.alpha, self.beta)))

    def observe(self, observation: Observation, reward: float) -> None:
        if not observation.valid:
            return
        # Keep a small prior mass after discounting to remain explorative.
        self.alpha = 1.0 + self.discount * (self.alpha - 1.0)
        self.beta = 1.0 + self.discount * (self.beta - 1.0)
        if observation.alert:
            self.alpha[observation.band] += 1.0
        else:
            self.beta[observation.band] += 1.0

    def predict_occupancy(self, band: int) -> Optional[float]:
        return float(self.alpha[band] / (self.alpha[band] + self.beta[band]))


class PredictiveGreedyScheduler(Scheduler):
    """A supervised-style exponentially smoothed occupancy predictor.

    It is a useful dependency-free reference for the optional predictive
    variant: labels are only received for the currently tuned band.
    """

    name = "predictive_greedy"

    def __init__(self, num_bands: int, smoothing: float = 0.15, exploration: float = 0.08, seed: int = 0) -> None:
        super().__init__(num_bands, seed)
        self.smoothing, self.exploration = float(smoothing), float(exploration)
        self.reset()

    def reset(self, seed: Optional[int] = None) -> None:
        super().reset(seed)
        self.probability = np.full(self.num_bands, 0.5, dtype=float)
        self.age = np.zeros(self.num_bands, dtype=int)

    def select_band(self) -> int:
        if self.rng.random() < self.exploration:
            return int(self.rng.integers(self.num_bands))
        # A small age bonus avoids permanently starving rarely sampled bands.
        score = self.probability + 0.03 * np.minimum(self.age, 20)
        return int(np.argmax(score))

    def observe(self, observation: Observation, reward: float) -> None:
        if not observation.valid:
            return
        self.age += 1
        band = observation.band
        self.age[band] = 0
        target = float(observation.alert)
        self.probability[band] = (1.0 - self.smoothing) * self.probability[band] + self.smoothing * target

    def predict_occupancy(self, band: int) -> Optional[float]:
        return float(self.probability[band])
