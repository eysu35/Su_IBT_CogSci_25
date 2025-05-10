import unittest
import numpy as np
from bandits import (
    stationary_MAB,
    drifting_MAB,
    stepwise_MAB,
    two_context_MAB,
    three_context_MAB,
    markovian_MAB,
    time_delayed_MAB,
)


class stepwise_MAB(stationary_MAB):
    def __init__(self, change_step=10):
        super().__init__()
        self.stepper = 0
        self.change = change_step

    def set_change_step(self, step):
        self.change = step

    def get_reward(self, a):
        # at step change, set the means of the first half of the arms to 0
        self.stepper += 1
        if self.stepper % self.change == 0:
            self.means = np.array(self.means, dtype=int)
            self.means[0 : self.narms // 2] = 0
        return np.random.normal(self.means[a], self.stds[a], 1)[0]


class two_context_MAB(stationary_MAB):
    def get_reward(self, a, c):
        # if state is falsy, flip the means
        if not c:
            self.means = -self.means
        return np.random.normal(self.means[a], self.stds[a], 1)[0]


class three_context_MAB(stationary_MAB):
    def get_reward(self, a, c):
        # if state 0, sort means ascending
        if c == 0:
            self.means = np.sort(self.means)
        # if state 1, sort means descending
        elif c == 1:
            self.means = np.sort(self.means)[::-1]
        # if state 2, randomly sort means
        else:
            np.random.shuffle(self.means)

        return np.random.normal(self.means[a], self.stds[a], 1)[0]


class markovian_MAB(stationary_MAB):
    def __init__(self):
        super().__init__()
        self.rewards = []

    def get_reward(self, a):
        r = np.random.normal(self.means[a], self.stds[a], 1)[0]
        self.rewards.append(r)
        # return average of previous rewards
        return np.mean(self.rewards)


class time_delayed_MAB(stationary_MAB):
    def __init__(self, delay=3):
        super().__init__()
        self.delay = delay
        self.rewards = []

    def get_reward(self, a):
        r = np.random.normal(self.means[a], self.stds[a], 1)[0]
        self.rewards.append(r)
        if len(self.rewards) < self.delay:
            return 0
        else:
            return self.rewards.pop(0)


class TestBandits(unittest.TestCase):
    def setUp(self):
        # Use a fixed seed for reproducibility in tests
        np.random.seed(42)
        self.means = [10, 20, 30, 40]
        self.stds = [1, 2, 3, 4]
        self.narms = len(self.means)

    def test_stationary_MAB(self):
        bandit = stationary_MAB()
        bandit.set_narms(self.narms)
        bandit.set_means(self.means)
        bandit.set_stds(self.stds)
        reward = bandit.get_reward(2)
        # Check reward is a float and near the specified mean (given randomness, we test type and rough range)
        self.assertIsInstance(reward, float)
        self.assertGreaterEqual(reward, self.means[2] - 10)
        self.assertLessEqual(reward, self.means[2] + 10)

    def test_drifting_MAB(self):
        bandit = drifting_MAB(drift_rate=2)
        bandit.set_narms(self.narms)
        bandit.set_means(self.means.copy())
        bandit.set_stds(self.stds)
        initial_means = bandit.means.copy()
        reward = bandit.get_reward(1)
        # The means should have drifted after one call.
        self.assertFalse(np.array_equal(bandit.means, initial_means))
        # Check reward is still a float.
        self.assertIsInstance(reward, float)

    def test_stepwise_MAB(self):
        change_step = 3  # Use a small change step for testing
        bandit = stepwise_MAB(change_step=change_step)
        bandit.set_narms(self.narms)
        bandit.set_means(self.means.copy())
        bandit.set_stds(self.stds)
        # Make calls fewer than change_step, means should remain unchanged.
        for _ in range(change_step - 1):
            _ = bandit.get_reward(0)
        self.assertTrue(np.array_equal(bandit.means, np.array(self.means)))
        # On the change_step-th call, first half means should be zeroed.
        _ = bandit.get_reward(0)
        expected_means = np.array(self.means)
        expected_means[: self.narms // 2] = 0
        self.assertTrue(np.array_equal(bandit.means, expected_means))

    def test_two_context_MAB(self):
        bandit = two_context_MAB()
        bandit.set_narms(self.narms)
        bandit.set_means(self.means.copy())
        bandit.set_stds(self.stds)
        # Call with a context that evaluates to False (e.g., 0 or False)
        reward_false = bandit.get_reward(1, c=False)
        # The means should have been negated
        self.assertTrue(np.array_equal(bandit.means, -np.array(self.means)))
        # Reset for a true context test.
        bandit.set_means(self.means.copy())
        reward_true = bandit.get_reward(1, c=True)
        # Means should not have flipped; still equal to the original means.
        self.assertTrue(np.array_equal(bandit.means, np.array(self.means)))
        self.assertIsInstance(reward_true, float)

    def test_three_context_MAB(self):
        bandit = three_context_MAB()
        bandit.set_narms(self.narms)
        original_means = np.array(self.means.copy())
        bandit.set_means(self.means.copy())
        bandit.set_stds(self.stds)
        # Context 0: ascending sort
        _ = bandit.get_reward(1, 0)
        self.assertTrue(np.array_equal(bandit.means, np.sort(original_means)))
        # Context 1: descending sort
        bandit.set_means(self.means.copy())
        _ = bandit.get_reward(1, 1)
        self.assertTrue(np.array_equal(bandit.means, np.sort(original_means)[::-1]))
        # Context 2: random shuffle (we cannot predict order, but ensure it's a permutation)
        bandit.set_means(self.means.copy())
        _ = bandit.get_reward(1, 2)
        self.assertCountEqual(bandit.means.tolist(), original_means.tolist())

    def test_markovian_MAB(self):
        bandit = markovian_MAB()
        bandit.set_narms(self.narms)
        bandit.set_means(self.means.copy())
        bandit.set_stds(self.stds)
        # Make a few calls and check that returned reward is the running average.
        rewards = []
        for i in range(5):
            r = bandit.get_reward(2)
            rewards.append(r)
            avg = np.mean(bandit.rewards)
            self.assertAlmostEqual(r, avg, places=5)

    def test_time_delayed_MAB(self):
        delay = 3
        bandit = time_delayed_MAB(delay=delay)
        bandit.set_narms(self.narms)
        bandit.set_means(self.means.copy())
        bandit.set_stds(self.stds)
        # First (delay-1) calls should return 0.
        rewards_collected = []
        for i in range(delay - 1):
            r = bandit.get_reward(0)
            rewards_collected.append(r)
            self.assertEqual(r, 0)
        # Next call should return the first generated reward.
        # We simulate one additional call.
        # To capture the first reward, we temporarily reinitialize to reset randomness.
        np.random.seed(42)
        # Compute what the first reward would be.
        expected_reward = np.random.normal(self.means[0], self.stds[0], 1)[0]
        # Make the call
        r_delayed = bandit.get_reward(0)
        self.assertAlmostEqual(r_delayed, expected_reward, places=5)


if __name__ == "__main__":
    unittest.main()
