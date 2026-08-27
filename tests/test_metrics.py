import unittest

from baselines import RoundRobinScheduler
from ew_env import EWEnvironment, EnvironmentConfig, PeriodicEmitter
from metrics import compute_metrics, run_episode


class MetricTests(unittest.TestCase):
    def test_perfect_single_band_receiver_has_pd_one(self) -> None:
        env = EWEnvironment(
            EnvironmentConfig(num_bands=1, detection_probability=1.0, false_alarm_probability=0.0),
            [PeriodicEmitter(0, hop_sequence=[0], dwell_slots=1, period_slots=2)],
        )
        trace = run_episode(env, RoundRobinScheduler(1), horizon=12, seed=5)
        result = compute_metrics(trace, env)
        self.assertEqual(result.probability_of_detection, 1.0)
        self.assertEqual(result.probability_false_alarm, 0.0)
        self.assertEqual(result.detected_emitters, 1)


if __name__ == "__main__":
    unittest.main()
