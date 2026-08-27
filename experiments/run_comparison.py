"""Run seeded policy comparisons and save a Markdown table, CSV, and plots.

Example:
    python -m experiments.run_comparison --episodes 20 --horizon 240 --sweep
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Callable, Dict, Iterable

import numpy as np

from baselines import (
    DiscountedUCBScheduler,
    PredictiveGreedyScheduler,
    RandomSweepScheduler,
    RoundRobinScheduler,
    Scheduler,
    StaticPriorityScheduler,
    ThompsonSamplingScheduler,
)
from ew_env import AgileEmitter, BurstyEmitter, EWEnvironment, EnvironmentConfig, PeriodicEmitter
from metrics import RunMetrics, average_metrics, comparison_table, evaluate
from periodic_intercept import RendezvousScheduler
from rl_scheduler import DQNConfig, DQNScheduler, train_dqn


def build_environment(num_bands: int = 8, agility: float = 0.85, seed: int = 11) -> EWEnvironment:
    """A mixed scenario containing periodic, agile, and bursty emitters."""
    hop = tuple(dict.fromkeys([0, num_bands // 3, (2 * num_bands) // 3]))
    emitters = [
        PeriodicEmitter(0, hop_sequence=hop, dwell_slots=2, period_slots=18, phase=1, priority=1.4, seed=seed + 1),
        AgileEmitter(1, active_probability=0.70, hop_probability=agility, priority=1.0, seed=seed + 2),
        BurstyEmitter(2, p_on=0.16, p_off=0.28, hop_probability=0.12 + 0.35 * agility, priority=0.8, seed=seed + 3),
    ]
    return EWEnvironment(
        EnvironmentConfig(
            num_bands=num_bands,
            detection_probability=0.91,
            false_alarm_probability=0.025,
            retune_cost=0.035,
            seed=seed,
        ),
        emitters,
    )


def policy_factories(num_bands: int, seed: int = 31) -> Dict[str, Callable[[], Scheduler]]:
    # Deliberately supplied prior; unlike learned policies it does not adapt.
    static_priorities = np.linspace(1.5, 0.5, num_bands)
    return {
        "round_robin": lambda: RoundRobinScheduler(num_bands, seed=seed),
        "random": lambda: RandomSweepScheduler(num_bands, seed=seed),
        "static_priority": lambda: StaticPriorityScheduler(num_bands, static_priorities, seed=seed),
        "discounted_ucb": lambda: DiscountedUCBScheduler(num_bands, seed=seed),
        "thompson": lambda: ThompsonSamplingScheduler(num_bands, seed=seed),
        "predictive": lambda: PredictiveGreedyScheduler(num_bands, seed=seed),
        "rendezvous": lambda: RendezvousScheduler(num_bands, max_period=32, seed=seed),
    }


def compare_methods(
    num_bands: int = 8,
    agility: float = 0.85,
    episodes: int = 16,
    horizon: int = 220,
    include_dqn: bool = True,
    seed: int = 500,
) -> Dict[str, RunMetrics]:
    """Train DQN once, then evaluate every method on common random seeds."""
    environment = build_environment(num_bands, agility, seed=seed)
    results = {
        name: average_metrics(evaluate(environment, factory, episodes, horizon, seed=seed + 100))
        for name, factory in policy_factories(num_bands, seed=seed + 7).items()
    }
    if include_dqn:
        dqn = DQNScheduler(num_bands, DQNConfig(), seed=seed + 19)
        train_dqn(environment, dqn, episodes=50, horizon=min(horizon, 180), seed=seed + 300)
        # The trained network is retained; reset clears only episode history.
        results["dqn"] = average_metrics(evaluate(environment, lambda: dqn, episodes, horizon, seed=seed + 100))
    return results


def _write_csv(results: Dict[str, RunMetrics], path: Path) -> None:
    rows = [{"method": name, **metrics.flat()} for name, metrics in results.items()]
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_comparison(results: Dict[str, RunMetrics], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Plot generation needs matplotlib; install requirements.txt first.") from error

    names = list(results)
    metrics = ["probability_of_detection", "probability_false_alarm", "intercept_rate", "average_reward"]
    labels = ["Pd", "Pfa", "Intercept rate", "Average reward"]
    figure, axes = plt.subplots(2, 2, figsize=(10, 6.5), constrained_layout=True)
    for axis, key, label in zip(axes.flat, metrics, labels):
        values = [getattr(results[name], key) for name in names]
        axis.bar(names, values, color="#3b82b6")
        axis.set_title(label)
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def sweep(
    output_dir: Path,
    band_counts: Iterable[int] = (4, 8, 12),
    agility_levels: Iterable[float] = (0.25, 0.85),
    episodes: int = 8,
    horizon: int = 180,
    seed: int = 900,
) -> None:
    """Create Pd-versus-band-count and agility comparison plots."""
    import matplotlib.pyplot as plt

    methods = ("round_robin", "discounted_ucb", "rendezvous")
    records = []
    for agility in agility_levels:
        for bands in band_counts:
            result = compare_methods(bands, agility, episodes, horizon, include_dqn=False, seed=seed + bands)
            for method in methods:
                records.append((agility, bands, method, result[method].probability_of_detection))
    with (output_dir / "sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("agility", "num_bands", "method", "Pd"))
        writer.writerows(records)
    figure, axes = plt.subplots(1, len(tuple(agility_levels)), figsize=(10, 4), sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for axis, agility in zip(axes, agility_levels):
        for method in methods:
            points = [(bands, pd) for a, bands, m, pd in records if a == agility and m == method]
            axis.plot([p[0] for p in points], [p[1] for p in points], marker="o", label=method)
        axis.set_title(f"agility={agility:.2f}")
        axis.set_xlabel("number of bands")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Pd")
    axes[-1].legend(fontsize=8)
    figure.savefig(output_dir / "band_agility_sweep.png", dpi=170)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs", help="directory for generated reports")
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=220)
    parser.add_argument("--bands", type=int, default=8)
    parser.add_argument("--agility", type=float, default=0.85)
    parser.add_argument("--no-dqn", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = compare_methods(args.bands, args.agility, args.episodes, args.horizon, not args.no_dqn)
    (output_dir / "summary.md").write_text(comparison_table(results) + "\n", encoding="utf-8")
    _write_csv(results, output_dir / "summary.csv")
    try:
        plot_comparison(results, output_dir / "comparison.png")
    except RuntimeError as error:
        print(f"Plot skipped: {error}")
    if args.sweep:
        sweep(output_dir, episodes=max(4, args.episodes // 2), horizon=args.horizon)
    print(comparison_table(results))
    print(f"\nSaved reports to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
