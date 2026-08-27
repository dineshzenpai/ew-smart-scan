import unittest

from baselines import DiscountedUCBScheduler, RoundRobinScheduler
from periodic_intercept import PeriodicityEstimator
from rl_scheduler import DQNConfig, DQNScheduler


class SchedulerTests(unittest.TestCase):
    def test_round_robin_cycles_every_band(self) -> None:
        scheduler = RoundRobinScheduler(3)
        scheduler.reset()
        self.assertEqual([scheduler.select_band() for _ in range(7)], [0, 1, 2, 0, 1, 2, 0])

    def test_ucb_explores_all_bands_before_revisiting(self) -> None:
        scheduler = DiscountedUCBScheduler(4)
        self.assertEqual([scheduler.select_band() for _ in range(4)], [0, 0, 0, 0])
        # A chosen arm is only marked seen after feedback, as it would be in a
        # real receiver control loop.
        scheduler.reset()
        from ew_env import Observation

        choices = []
        for time in range(4):
            band = scheduler.select_band()
            choices.append(band)
            scheduler.observe(Observation(time, band, False, False), 0.0)
        self.assertEqual(choices, [0, 1, 2, 3])

    def test_periodicity_estimator_recovers_a_synthetic_cycle(self) -> None:
        estimator = PeriodicityEstimator(min_period=6, max_period=18, min_observations=12)
        for time in range(96):
            estimator.observe(time, time % 12 in (2, 3, 4))
        estimate = estimator.estimate()
        self.assertIsNotNone(estimate)
        assert estimate is not None
        self.assertEqual(estimate.period, 12)
        self.assertGreaterEqual(estimate.dwell, 2)

    def test_dqn_state_has_expected_shape(self) -> None:
        scheduler = DQNScheduler(5, DQNConfig(history_length=4, hidden_size=8, warmup_steps=2, batch_size=2))
        self.assertEqual(scheduler._state().shape, (35,))


if __name__ == "__main__":
    unittest.main()
