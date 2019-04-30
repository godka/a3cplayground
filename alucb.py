import numpy as np
ACTION_DIMS = 10 + 1

class AutoLearn(object):

    def __init__(self, learning_rate):
        self.a_dims = ACTION_DIMS
        self.lr = learning_rate
        self.lr_vec = np.linspace(self.lr * 10, self.lr / 10., ACTION_DIMS)
        self.true_rewards = np.random.uniform(low=0, high=1, size=self.a_dims)
        self.estimated_rewards = np.zeros(self.a_dims)
        self.chosen_count = np.zeros(self.a_dims)
        self.total_reward = 0.
        self.iter = 1.

    def calculate_delta(self, T, item):
        if self.chosen_count[item] == 0:
            return 1
        else:
            return np.sqrt(2 * np.log(T) / self.chosen_count[item])

    def update(self, rew):
        t = self.iter
        item = self.a_last
        self.estimated_rewards[item] = ((t - 1) * self.estimated_rewards[item] + rew) / t
        self.chosen_count[item] += 1
        self.iter += 1

    def select(self):
        t = self.iter
        upper_bound_probs = [self.estimated_rewards[item] + self.calculate_delta(t, item) for item in range(self.a_dims)]
        item = np.argmax(upper_bound_probs)
        #reward = np.random.binomial(n=1, p=self.true_rewards[item])
        self.a_last = item
        return item
