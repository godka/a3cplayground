import numpy as np
ACTION_DIMS = 10 + 1
GAMMA = 0.99
EPS = 0.2


class AutoLearn(object):

    def __init__(self, learning_rate):
        self.a_dims = ACTION_DIMS
        self.lr = learning_rate
        self.lr_vec = np.linspace(self.lr * 10, self.lr / 10., ACTION_DIMS)
        self.a_vecs = np.zeros(self.a_dims)
        self.a_last = np.zeros(self.a_dims)
        # self.a_vecs.fill(0.)
        self.a_last.fill(-1000000.)
        self.last_act = None

    def update(self, rew):
        self.a_last *= GAMMA
        self.a_last[self.last_act] = rew
        self.a_vecs *= GAMMA
        self.a_vecs += self.a_last * (1-GAMMA)
        # self.a_dims[action] += rew * (1 - GAMMA)

    def select(self):
        _act = -1
        for i, p in enumerate(self.a_last):
            if p == -1000000:
                _act = i
                break
        if _act < 0:
            if np.random.random() < EPS:
                _act = np.random.randint(self.a_dims)
            else:
                _act = np.argmax(self.a_vecs)
        self.last_act = _act
        return self.lr_vec[_act]
