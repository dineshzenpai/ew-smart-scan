"""A small, dependency-free DQN scheduler for the partially observed receiver.

This is intentionally a research baseline rather than an optimized deep-RL
library.  It implements experience replay, a target network, epsilon-greedy
exploration, and a state made solely from receiver-observable history.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import numpy as np

from baselines import Scheduler
from ew_env import EWEnvironment, Observation


@dataclass(frozen=True)
class DQNConfig:
    history_length: int = 6
    hidden_size: int = 48
    learning_rate: float = 0.003
    gamma: float = 0.96
    replay_capacity: int = 8_000
    batch_size: int = 48
    warmup_steps: int = 96
    target_sync_interval: int = 100
    epsilon_start: float = 0.70
    epsilon_end: float = 0.04
    epsilon_decay_steps: int = 3_000


class DQNScheduler(Scheduler):
    """A NumPy DQN policy over bands, with no access to environment truth."""

    name = "dqn"

    def __init__(self, num_bands: int, config: DQNConfig = DQNConfig(), seed: int = 0) -> None:
        super().__init__(num_bands, seed)
        self.config = config
        self.feature_size = num_bands * (config.history_length + 3)
        init_rng = np.random.default_rng(seed)
        scale1 = np.sqrt(2.0 / self.feature_size)
        scale2 = np.sqrt(2.0 / config.hidden_size)
        self.w1 = init_rng.normal(0.0, scale1, (self.feature_size, config.hidden_size))
        self.b1 = np.zeros(config.hidden_size)
        self.w2 = init_rng.normal(0.0, scale2, (config.hidden_size, num_bands))
        self.b2 = np.zeros(num_bands)
        self.target_w1, self.target_b1 = self.w1.copy(), self.b1.copy()
        self.target_w2, self.target_b2 = self.w2.copy(), self.b2.copy()
        self.replay: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=config.replay_capacity)
        self.training = True
        self.global_steps = 0
        self.reset()

    def reset(self, seed: Optional[int] = None) -> None:
        super().reset(seed)
        self.history = np.zeros((self.num_bands, self.config.history_length), dtype=float)
        self.age = np.zeros(self.num_bands, dtype=float)
        self.current_band: Optional[int] = None
        self._last_state: Optional[np.ndarray] = None
        self._last_action: Optional[int] = None

    def set_training(self, enabled: bool) -> None:
        self.training = bool(enabled)

    def _state(self) -> np.ndarray:
        # Recent alerts/clears (-1/1, zeros for unobserved), time since visit,
        # current-band one-hot, and switch-cost readiness form the POMDP state.
        history = self.history.reshape(-1)
        age = np.minimum(self.age, 30.0) / 30.0
        current = np.zeros(self.num_bands)
        if self.current_band is not None:
            current[self.current_band] = 1.0
        switch = np.full(self.num_bands, 1.0 if self.current_band is not None else 0.0)
        return np.concatenate((history, age, current, switch)).astype(float)

    @staticmethod
    def _forward(x: np.ndarray, w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        hidden = np.maximum(0.0, x @ w1 + b1)
        return hidden @ w2 + b2, hidden

    def _q_values(self, state: np.ndarray, target: bool = False) -> np.ndarray:
        if target:
            q, _ = self._forward(state, self.target_w1, self.target_b1, self.target_w2, self.target_b2)
        else:
            q, _ = self._forward(state, self.w1, self.b1, self.w2, self.b2)
        return q

    def _epsilon(self) -> float:
        fraction = min(1.0, self.global_steps / max(1, self.config.epsilon_decay_steps))
        return self.config.epsilon_start + fraction * (self.config.epsilon_end - self.config.epsilon_start)

    def select_band(self) -> int:
        state = self._state()
        if self.training and self.rng.random() < self._epsilon():
            action = int(self.rng.integers(self.num_bands))
        else:
            action = int(np.argmax(self._q_values(state)))
        self._last_state, self._last_action = state, action
        return action

    def observe(self, observation: Observation, reward: float) -> None:
        self.age += 1.0
        band = observation.band
        if observation.valid:
            self.age[band] = 0.0
            self.history[band, :-1] = self.history[band, 1:]
            self.history[band, -1] = 1.0 if observation.alert else -1.0
        self.current_band = band
        next_state = self._state()
        if self._last_state is not None and self._last_action is not None:
            self.replay.append((self._last_state, self._last_action, float(reward), next_state, False))
        if self.training:
            self.global_steps += 1
            self._learn()

    def _learn(self) -> None:
        if len(self.replay) < max(self.config.warmup_steps, self.config.batch_size):
            return
        indices = self.rng.choice(len(self.replay), size=self.config.batch_size, replace=False)
        batch = [self.replay[int(index)] for index in indices]
        states = np.stack([item[0] for item in batch])
        actions = np.asarray([item[1] for item in batch])
        rewards = np.asarray([item[2] for item in batch])
        next_states = np.stack([item[3] for item in batch])
        done = np.asarray([item[4] for item in batch])

        q, hidden = self._forward(states, self.w1, self.b1, self.w2, self.b2)
        target_next, _ = self._forward(next_states, self.target_w1, self.target_b1, self.target_w2, self.target_b2)
        td_target = rewards + self.config.gamma * (1.0 - done) * np.max(target_next, axis=1)
        td_error = q[np.arange(len(batch)), actions] - td_target
        grad_q = np.zeros_like(q)
        grad_q[np.arange(len(batch)), actions] = 2.0 * td_error / len(batch)
        grad_w2 = hidden.T @ grad_q
        grad_b2 = grad_q.sum(axis=0)
        grad_hidden = grad_q @ self.w2.T
        grad_hidden[hidden <= 0.0] = 0.0
        grad_w1 = states.T @ grad_hidden
        grad_b1 = grad_hidden.sum(axis=0)
        lr = self.config.learning_rate
        self.w1 -= lr * np.clip(grad_w1, -5.0, 5.0)
        self.b1 -= lr * np.clip(grad_b1, -5.0, 5.0)
        self.w2 -= lr * np.clip(grad_w2, -5.0, 5.0)
        self.b2 -= lr * np.clip(grad_b2, -5.0, 5.0)
        if self.global_steps % self.config.target_sync_interval == 0:
            self.target_w1, self.target_b1 = self.w1.copy(), self.b1.copy()
            self.target_w2, self.target_b2 = self.w2.copy(), self.b2.copy()

    def predict_occupancy(self, band: int) -> Optional[float]:
        # A transparent proxy rather than a claim that Q-values are calibrated
        # probabilities; it supports comparable prediction-accuracy logging.
        recent = self.history[band]
        observed = recent[recent != 0]
        if not len(observed):
            return None
        return float(np.clip((observed.mean() + 1.0) / 2.0, 0.0, 1.0))


def train_dqn(
    environment: EWEnvironment,
    agent: DQNScheduler,
    episodes: int = 80,
    horizon: int = 180,
    seed: int = 1_000,
) -> list[float]:
    """Train *agent* in repeated seeded episodes and return episode rewards."""
    if environment.config.num_bands != agent.num_bands:
        raise ValueError("agent and environment must have the same band count")
    rewards: list[float] = []
    agent.set_training(True)
    for episode in range(episodes):
        environment.reset(seed + episode)
        agent.reset(seed + episode)
        total = 0.0
        for _ in range(horizon):
            result = environment.step(agent.select_band())
            agent.observe(result.observation, result.reward)
            total += result.reward
        rewards.append(float(total))
    agent.set_training(False)
    return rewards
