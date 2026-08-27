import unittest

from ew_env import EWEnvironment, EnvironmentConfig, PeriodicEmitter


class EnvironmentTests(unittest.TestCase):
    def test_periodic_emitter_uses_dwell_then_quiet_interval(self) -> None:
        emitter = PeriodicEmitter(3, hop_sequence=[0, 1], dwell_slots=2, period_slots=8)
        env = EWEnvironment(EnvironmentConfig(num_bands=2, detection_probability=1.0, false_alarm_probability=0.0), [emitter])
        env.reset(4)
        observed_truth = []
        for _ in range(8):
            observed_truth.append(env.step(0).truth.occupied_bands)
        self.assertEqual(observed_truth[:2], [(True, False), (True, False)])
        self.assertEqual(observed_truth[2:4], [(False, True), (False, True)])
        self.assertEqual(observed_truth[4:], [(False, False)] * 4)

    def test_perfect_sensor_has_no_false_alarms(self) -> None:
        emitter = PeriodicEmitter(0, hop_sequence=[0], dwell_slots=1, period_slots=2)
        env = EWEnvironment(EnvironmentConfig(num_bands=2, detection_probability=1.0, false_alarm_probability=0.0), [emitter])
        env.reset(8)
        active = env.step(0)
        quiet = env.step(1)
        self.assertTrue(active.true_detection)
        self.assertFalse(active.false_alarm)
        self.assertFalse(quiet.true_detection)
        self.assertFalse(quiet.false_alarm)

    def test_retune_slots_produce_blind_dwell(self) -> None:
        env = EWEnvironment(
            EnvironmentConfig(num_bands=2, detection_probability=1.0, false_alarm_probability=0.0, retune_slots=1),
            [PeriodicEmitter(0, hop_sequence=[1], dwell_slots=4, period_slots=4)],
        )
        env.reset(2)
        env.step(0)  # Initial tune is immediately usable.
        blind = env.step(1)
        detected = env.step(1)
        self.assertFalse(blind.observation.valid)
        self.assertFalse(blind.true_detection)
        self.assertTrue(detected.observation.valid)
        self.assertTrue(detected.true_detection)


if __name__ == "__main__":
    unittest.main()
