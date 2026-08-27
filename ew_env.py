"""Discrete-time electronic-support (ES) receiver simulation.

The simulation deliberately keeps the RF physics abstract.  An emitter can
occupy one of a receiver's discrete frequency bands in a time slot and the
receiver can observe only the band selected for that slot.  Ground truth is
returned for experiment logging, but schedulers are given only ``Observation``
objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class EnvironmentConfig:
    """Configuration shared by a receiver and all simulated emitters."""

    num_bands: int = 8
    detection_probability: float = 0.92
    false_alarm_probability: float = 0.02
    retune_cost: float = 0.05
    retune_slots: int = 0
    false_alarm_penalty: float = 0.25
    repeat_detection_penalty: float = 0.10
    slot_duration: float = 1.0
    seed: int = 7

    def __post_init__(self) -> None:
        if self.num_bands < 1:
            raise ValueError("num_bands must be positive")
        for name in ("detection_probability", "false_alarm_probability"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.retune_slots < 0:
            raise ValueError("retune_slots must be non-negative")


@dataclass(frozen=True)
class Observation:
    """The only sensor feedback made available to scheduling policies."""

    time: int
    band: int
    alert: bool
    switched: bool
    valid: bool = True


@dataclass(frozen=True)
class GroundTruth:
    """Diagnostic state for scoring only; do not pass it to a scheduler."""

    time: int
    occupied_bands: Tuple[bool, ...]
    emitters_by_band: Dict[int, Tuple[int, ...]]
    emission_starts: Dict[int, int]


@dataclass(frozen=True)
class StepResult:
    observation: Observation
    reward: float
    true_detection: bool
    false_alarm: bool
    truth: GroundTruth


class Emitter:
    """Base class for an emitter with an independently seeded process."""

    emitter_type = "base"

    def __init__(self, emitter_id: int, priority: float = 1.0, seed: int = 0) -> None:
        self.emitter_id = emitter_id
        self.priority = float(priority)
        self.seed = int(seed)
        self._rng = np.random.default_rng(seed)
        self._active = False
        self._band: Optional[int] = None
        self._started_at: Optional[int] = None

    def reset(self, seed: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(self.seed if seed is None else seed)
        self._active, self._band, self._started_at = False, None, None

    def state_at(self, time: int, num_bands: int) -> Tuple[Optional[int], Optional[int]]:
        """Return ``(band, emission_start)`` for *time* (implemented by children)."""
        raise NotImplementedError

    def _set_state(self, active: bool, band: Optional[int], time: int) -> Tuple[Optional[int], Optional[int]]:
        if active:
            if not self._active or band != self._band:
                self._started_at = time
            self._active, self._band = True, band
            return band, self._started_at
        self._active, self._band, self._started_at = False, None, None
        return None, None


class PeriodicEmitter(Emitter):
    """A periodic scanner with a fixed hop sequence and optional quiet period."""

    emitter_type = "periodic"

    def __init__(
        self,
        emitter_id: int,
        hop_sequence: Sequence[int],
        dwell_slots: int = 2,
        period_slots: int = 16,
        phase: int = 0,
        priority: float = 1.0,
        seed: int = 0,
    ) -> None:
        super().__init__(emitter_id, priority, seed)
        if not hop_sequence or dwell_slots < 1 or period_slots < dwell_slots * len(hop_sequence):
            raise ValueError("period must accommodate a non-empty hop sequence")
        self.hop_sequence = tuple(int(b) for b in hop_sequence)
        self.dwell_slots = int(dwell_slots)
        self.period_slots = int(period_slots)
        self.phase = int(phase)

    def state_at(self, time: int, num_bands: int) -> Tuple[Optional[int], Optional[int]]:
        if any(b < 0 or b >= num_bands for b in self.hop_sequence):
            raise ValueError("hop sequence contains a band outside the environment")
        cycle_position = (time + self.phase) % self.period_slots
        active_length = self.dwell_slots * len(self.hop_sequence)
        if cycle_position >= active_length:
            return self._set_state(False, None, time)
        index = cycle_position // self.dwell_slots
        band = self.hop_sequence[index]
        # This start is analytical so it remains correct if the same band
        # appears more than once in a hop sequence.
        start = time - (cycle_position % self.dwell_slots)
        self._active, self._band, self._started_at = True, band, start
        return band, start


class AgileEmitter(Emitter):
    """A pseudo-random hopper, optionally silent in a fraction of slots."""

    emitter_type = "agile"

    def __init__(
        self,
        emitter_id: int,
        active_probability: float = 0.75,
        hop_probability: float = 0.85,
        priority: float = 1.0,
        seed: int = 0,
    ) -> None:
        super().__init__(emitter_id, priority, seed)
        self.active_probability = float(active_probability)
        self.hop_probability = float(hop_probability)

    def state_at(self, time: int, num_bands: int) -> Tuple[Optional[int], Optional[int]]:
        active = bool(self._rng.random() < self.active_probability)
        if active:
            if self._band is None or self._rng.random() < self.hop_probability:
                band = int(self._rng.integers(num_bands))
            else:
                band = self._band
            return self._set_state(True, band, time)
        return self._set_state(False, None, time)


class BurstyEmitter(Emitter):
    """A two-state Markov on/off emitter with optional hopping while active."""

    emitter_type = "bursty"

    def __init__(
        self,
        emitter_id: int,
        p_on: float = 0.18,
        p_off: float = 0.32,
        hop_probability: float = 0.10,
        priority: float = 1.0,
        seed: int = 0,
    ) -> None:
        super().__init__(emitter_id, priority, seed)
        self.p_on, self.p_off, self.hop_probability = float(p_on), float(p_off), float(hop_probability)

    def state_at(self, time: int, num_bands: int) -> Tuple[Optional[int], Optional[int]]:
        if self._active:
            active = not bool(self._rng.random() < self.p_off)
        else:
            active = bool(self._rng.random() < self.p_on)
        if active:
            if self._band is None or self._rng.random() < self.hop_probability:
                band = int(self._rng.integers(num_bands))
            else:
                band = self._band
            return self._set_state(True, band, time)
        return self._set_state(False, None, time)


class EWEnvironment:
    """Partially observable RF environment with a single tunable receiver."""

    def __init__(self, config: EnvironmentConfig, emitters: Iterable[Emitter]) -> None:
        self.config = config
        self.emitters = list(emitters)
        ids = [emitter.emitter_id for emitter in self.emitters]
        if len(ids) != len(set(ids)):
            raise ValueError("emitter_id values must be unique")
        self._rng = np.random.default_rng(config.seed)
        self.time = 0
        self.current_band: Optional[int] = None
        self._retune_remaining = 0
        self.detected_emitters: set[int] = set()

    def reset(self, seed: Optional[int] = None) -> None:
        base_seed = self.config.seed if seed is None else int(seed)
        self._rng = np.random.default_rng(base_seed)
        self.time, self.current_band, self._retune_remaining = 0, None, 0
        self.detected_emitters = set()
        for index, emitter in enumerate(self.emitters):
            emitter.reset(base_seed + 10_007 * (index + 1))

    def _truth(self) -> GroundTruth:
        by_band: Dict[int, List[int]] = {band: [] for band in range(self.config.num_bands)}
        starts: Dict[int, int] = {}
        for emitter in self.emitters:
            band, start = emitter.state_at(self.time, self.config.num_bands)
            if band is not None:
                by_band[band].append(emitter.emitter_id)
                starts[emitter.emitter_id] = int(start if start is not None else self.time)
        return GroundTruth(
            time=self.time,
            occupied_bands=tuple(bool(by_band[b]) for b in range(self.config.num_bands)),
            emitters_by_band={band: tuple(ids) for band, ids in by_band.items() if ids},
            emission_starts=starts,
        )

    def step(self, band: int) -> StepResult:
        """Tune to *band* for one dwell slot and return a sensor result.

        The returned ``truth`` is for an evaluator.  A controller should use
        only ``result.observation`` and ``result.reward``.
        """
        if not 0 <= band < self.config.num_bands:
            raise ValueError(f"band must be in [0, {self.config.num_bands - 1}]")
        truth = self._truth()
        switched = self.current_band is not None and self.current_band != band
        if switched:
            self._retune_remaining = self.config.retune_slots
        valid = self._retune_remaining == 0
        occupied = truth.occupied_bands[band]
        if not valid:
            alert = False
            self._retune_remaining -= 1
        elif occupied:
            alert = bool(self._rng.random() < self.config.detection_probability)
        else:
            alert = bool(self._rng.random() < self.config.false_alarm_probability)
        true_detection = bool(valid and occupied and alert)
        false_alarm = bool(valid and not occupied and alert)
        reward = -self.config.retune_cost if switched else 0.0
        if true_detection:
            emitter_map = {e.emitter_id: e for e in self.emitters}
            reward += max(emitter_map[eid].priority for eid in truth.emitters_by_band[band])
            newly_found = set(truth.emitters_by_band[band]) - self.detected_emitters
            if not newly_found:
                reward -= self.config.repeat_detection_penalty
            self.detected_emitters.update(truth.emitters_by_band[band])
        elif false_alarm:
            reward -= self.config.false_alarm_penalty
        result = StepResult(
            observation=Observation(self.time, band, alert, switched, valid),
            reward=float(reward),
            true_detection=true_detection,
            false_alarm=false_alarm,
            truth=truth,
        )
        self.current_band = band
        self.time += 1
        return result
