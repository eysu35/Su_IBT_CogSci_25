import numpy as np


# stationary multi-armed bandit
class stationary_MAB:
    def __init__(self):
        self.narms = None
        self.means = None
        self.stds = None

    def set_narms(self, n):
        self.narms = n

    def set_means(self, arms):
        self.means = np.array(arms, dtype=int)

    def set_stds(self, stds):
        self.stds = np.array(stds, dtype=int)

    def get_reward(self, a):
        return np.random.normal(self.means[a], self.stds[a], 1)[0]


######## ALL REMAINING BANDITS INHERIT FROM STATIONARY MAB ########


# non-stationary multi-armed bandit (drifting)
class drifting_MAB(stationary_MAB):
    def __init__(self, drift_rate=2):
        super().__init__()
        self.drift_rate = drift_rate

    def set_drift_rate(self, rate):
        self.drift_rate = rate

    def get_reward(self, a):
        self.means = self.means.astype(float)
        self.means += np.random.normal(0, self.drift_rate, self.narms)
        return np.random.normal(self.means[a], self.stds[a], 1)[0]


# non-stationary multi-armed bandit (stepwise)
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


# 2-context multi-armed bandit
class two_context_MAB(stationary_MAB):
    def get_reward(self, a, c):
        # if state 2, flip the means
        if not c:
            self.means = -self.means
        return np.random.normal(self.means[a], self.stds[a], 1)[0]


# 3-context multi-armed bandit
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


# moving average multi-armed bandit
class moving_avg_MAB(stationary_MAB):
    def __init__(self):
        super().__init__()
        self.rewards = []

    def get_reward(self, a):
        r = np.random.normal(self.means[a], self.stds[a], 1)[0]
        self.rewards.append(r)
        # return average of previous rewards
        return np.mean(self.rewards)


# time-delayed multi-armed bandit
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


def main():
    # instantiate and test the bandit classes by getting 20 rewards
    print("stationary")
    bandit = stationary_MAB()
    bandit.set_narms(4)
    bandit.set_means([10, 20, 30, 40])
    bandit.set_stds([4, 4, 4, 4])
    for i in range(20):
        print(bandit.get_reward(i % 4))
        print(bandit.means)

    print("\ndrifting")
    bandit = drifting_MAB()
    bandit.set_narms(4)
    bandit.set_means([10, 20, 30, 40])
    bandit.set_stds([4, 4, 4, 4])
    for i in range(20):
        print(bandit.get_reward(i % 4))
        print(bandit.means)

    print("\nstepwise")
    bandit = stepwise_MAB()
    bandit.set_narms(4)
    bandit.set_means([10, 20, 30, 40])
    bandit.set_stds([4, 4, 4, 4])
    for i in range(20):
        print(bandit.get_reward(i % 4))
        print(bandit.means)

    print("\nmoving average")
    bandit = moving_avg_MAB()
    bandit.set_narms(4)
    bandit.set_means([10, 20, 30, 40])
    bandit.set_stds([4, 4, 4, 4])
    for i in range(20):
        print(bandit.get_reward(i % 4))
        print(bandit.means)

    print("\ntime delayed")
    bandit = time_delayed_MAB()
    bandit.set_narms(4)
    bandit.set_means([10, 20, 30, 40])
    bandit.set_stds([4, 4, 4, 4])
    for i in range(20):
        print(bandit.get_reward(i % 4))
        print(bandit.means)


if __name__ == "__main__":
    main()
