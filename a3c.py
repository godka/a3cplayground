import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tl
import time

GAMMA = 0.99
ENTROPY_WEIGHT = 0.1
ENTROPY_EPS = 1e-6
S_INFO = 4


class RelayBuffer(object):
    def __init__(self, max_num):
        self.max_num = max_num
        self.buffer = []

    def push(self, s, a, r, t, m_a):
        # for _s, _a, _r, _m_a in zip(s, a, r, m_a):
        if len(self.buffer) == self.max_num:
            rnd_pop = np.random.randint(self.max_num)
            self.buffer.pop(rnd_pop)
        self.buffer.append((s, a, r, t, m_a))

    def pull(self):
        np.random.shuffle(self.buffer)
        _s, _a, _r, _t, _m_a = self.buffer[0]
        return _s, _a, _r, _t, _m_a


class ActorNetwork(object):
    """
    Input to the network is the state, output is the distribution
    of all actions.
    """

    def __init__(self, sess, state_dim, action_dim, learning_rate):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr_rate = learning_rate
        #self.replay_buffer = RelayBuffer()

        # Create the actor network
        self.inputs = tf.placeholder(tf.float32, [None, self.s_dim])
        self.out = self.create_actor_network()

        # Get all network parameters
        self.network_params = \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')

        # Set all network parameters
        self.input_network_params = []
        for param in self.network_params:
            self.input_network_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        self.set_network_params_op = []
        for idx, param in enumerate(self.input_network_params):
            self.set_network_params_op.append(
                self.network_params[idx].assign(param))

        # Selected action, 0-1 vector
        self.acts = tf.placeholder(tf.float32, [None, self.a_dim])
        # stop gradient now
        self.mu_acts = tf.placeholder(tf.float32, [None, self.a_dim])
        self.real_out = tf.clip_by_value(self.out, 1e-4, 1.)
        self.mu_weight = tf.stop_gradient(tf.clip_by_value(tf.reduce_sum(tf.multiply(
            self.acts, self.real_out / self.mu_acts), reduction_indices=1, keep_dims=True), 0., 1.))
        #tf.placeholder(tf.float32, [None, 1])

        # This gradient will be provided by the critic network
        self.act_grad_weights = tf.placeholder(tf.float32, [None, 1])

        # Compute the objective (log action_vector and entropy)
        # self.obj = tf.reduce_sum(tf.multiply(
        #                tf.log(tf.reduce_sum(tf.multiply(self.out, self.acts),
        #                                     reduction_indices=1, keep_dims=True)),
        #                -self.act_grad_weights)) \
        #            + ENTROPY_WEIGHT * tf.reduce_sum(tf.multiply(self.out,
        #                                                    tf.log(self.out + ENTROPY_EPS)))
        # Compute the objective (log action_vector and entropy)
        self.obj = tf.reduce_sum(
            tf.multiply(
                self.mu_weight * tf.log(tf.reduce_sum(tf.multiply(self.real_out, self.acts),
                                                      reduction_indices=1, keep_dims=True)),
                -self.act_grad_weights)) \
            + ENTROPY_WEIGHT * tf.reduce_sum(tf.multiply(self.real_out,
                                                         tf.log(self.real_out + ENTROPY_EPS)))

        # Combine the gradients here
        self.actor_gradients = tf.gradients(self.obj, self.network_params)

        # Optimization Op
        self.optimize = tf.train.RMSPropOptimizer(self.lr_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.opt = tf.train.RMSPropOptimizer(self.lr_rate).minimize(self.obj)

    def create_actor_network(self):
        with tf.variable_scope('actor'):
            hid_1 = tl.fully_connected(
                self.inputs, 32, activation_fn=tf.nn.softplus)
            hid_2 = tl.fully_connected(hid_1, 16, activation_fn=tf.nn.softplus)
            hid_3 = tl.fully_connected(hid_2, 8, activation_fn=tf.nn.softplus)
            out = tl.fully_connected(
                hid_3, self.a_dim, activation_fn=tf.nn.softmax)
            return out

    def train(self, inputs, acts, act_grad_weights):

        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.acts: acts,
            self.act_grad_weights: act_grad_weights
        })

    def predict(self, inputs):
        return self.sess.run(self.real_out, feed_dict={
            self.inputs: inputs
        })

    def get_gradients(self, inputs, acts, act_grad_weights, mu_acts):
        # _acts = self.predict(inputs)
        # # _mu_weight = self.acts * _acts / mu_acts
        # print(mu_acts, _acts, acts)
        # # print(self.sess.run(self.mu_weight, feed_dict={
        # #     self.inputs: inputs,
        # #     self.mu_acts: mu_acts,
        # #     self.acts: acts,
        # # }))
        return self.sess.run(self.actor_gradients, feed_dict={
            self.inputs: inputs,
            self.mu_acts: mu_acts,
            self.acts: acts,
            self.act_grad_weights: act_grad_weights
        })

    def train_v2(self, inputs, acts, act_grad_weights, mu_acts):
        # _acts = self.predict(inputs)
        # # _mu_weight = self.acts * _acts / mu_acts
        # print(mu_acts, _acts, acts)
        # # print(self.sess.run(self.mu_weight, feed_dict={
        # #     self.inputs: inputs,
        # #     self.mu_acts: mu_acts,
        # #     self.acts: acts,
        # # }))
        return self.sess.run(self.opt, feed_dict={
            self.inputs: inputs,
            self.mu_acts: mu_acts,
            self.acts: acts,
            self.act_grad_weights: act_grad_weights
        })

    def apply_gradients(self, actor_gradients):
        return self.sess.run(self.optimize, feed_dict={
            i: d for i, d in zip(self.actor_gradients, actor_gradients)
        })

    def get_network_params(self):
        return self.sess.run(self.network_params)

    def set_network_params(self, input_network_params):
        self.sess.run(self.set_network_params_op, feed_dict={
            i: d for i, d in zip(self.input_network_params, input_network_params)
        })


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is V(s).
    On policy: the action must be obtained from the output of the Actor network.
    """

    def __init__(self, sess, state_dim, learning_rate):
        self.sess = sess
        self.s_dim = state_dim
        self.lr_rate = learning_rate

        # Create the critic network
        self.inputs = tf.placeholder(tf.float32, [None, self.s_dim])
        self.out = self.create_critic_network()

        # Get all network parameters
        self.network_params = \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')

        # Set all network parameters
        self.input_network_params = []
        for param in self.network_params:
            self.input_network_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        self.set_network_params_op = []
        for idx, param in enumerate(self.input_network_params):
            self.set_network_params_op.append(
                self.network_params[idx].assign(param))

        # Network target V(s)
        self.td_target = tf.placeholder(tf.float32, [None, 1])

        # Temporal Difference, will also be weights for actor_gradients
        self.td = tf.subtract(self.td_target, self.out)

        # Mean square error
        self.loss = tf.reduce_mean(tf.square(self.td_target - self.out))

        # Compute critic gradient
        self.critic_gradients = tf.gradients(self.loss, self.network_params)

        # Optimization Op
        self.optimize = tf.train.RMSPropOptimizer(self.lr_rate).\
            apply_gradients(zip(self.critic_gradients, self.network_params))

        self.opt = tf.train.RMSPropOptimizer(self.lr_rate).minimize(self.loss)

    def create_critic_network(self):
        with tf.variable_scope('critic'):
            hid_1 = tl.fully_connected(
                self.inputs, 32, activation_fn=tf.nn.softplus)
            hid_2 = tl.fully_connected(hid_1, 16, activation_fn=tf.nn.softplus)
            hid_3 = tl.fully_connected(hid_2, 8, activation_fn=tf.nn.softplus)
            out = tl.fully_connected(hid_3, 1, activation_fn=None)

            return out

    def train(self, inputs, td_target):
        return self.sess.run([self.loss, self.optimize], feed_dict={
            self.inputs: inputs,
            self.td_target: td_target
        })

    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs
        })

    def get_td(self, inputs, td_target):
        return self.sess.run(self.td, feed_dict={
            self.inputs: inputs,
            self.td_target: td_target
        })

    def get_gradients(self, inputs, td_target):
        return self.sess.run(self.critic_gradients, feed_dict={
            self.inputs: inputs,
            self.td_target: td_target
        })

    def train_v2(self, inputs, td_target):
        return self.sess.run(self.opt, feed_dict={
            self.inputs: inputs,
            self.td_target: td_target
        })

    def apply_gradients(self, critic_gradients):
        return self.sess.run(self.optimize, feed_dict={
            i: d for i, d in zip(self.critic_gradients, critic_gradients)
        })

    def get_network_params(self):
        return self.sess.run(self.network_params)

    def set_network_params(self, input_network_params):
        self.sess.run(self.set_network_params_op, feed_dict={
            i: d for i, d in zip(self.input_network_params, input_network_params)
        })

def train_v2(s_batch, a_batch, r_batch, terminal, actor, critic, mu_a_batch=None):
    """
    batch of s, a, r is from samples in a sequence
    the format is in np.array([batch_size, s/a/r_dim])
    terminal is True when sequence ends as a terminal state
    """
    assert s_batch.shape[0] == a_batch.shape[0]
    assert s_batch.shape[0] == r_batch.shape[0]
    ba_size = s_batch.shape[0]

    v_batch = critic.predict(s_batch)
    probe_batch = actor.predict(s_batch)
    R_batch = np.zeros(r_batch.shape)
    V_batch = np.zeros(r_batch.shape)
    # impala
    clipped_rhos = []
    for i in range(ba_size):
        a = np.argmax(a_batch[i])
        clipped_rhos.append(
            np.clip(probe_batch[i][a] / mu_a_batch[i][a], 0., 1.))

    # clipped_rhos = tf.minimum(clip_rho_threshold, rhos, name='clipped_rhos')
    if terminal:
        R_batch[-1, 0] = 0  # terminal state
    else:
        R_batch[-1, 0] = v_batch[-1, 0]  # boot strap from last state

    for t in reversed(range(ba_size - 1)):
        R_batch[t, 0] = r_batch[t] + GAMMA * R_batch[t + 1, 0]

    td_batch = R_batch - v_batch
    # impala
    for t in reversed(range(ba_size - 1)):
        V_batch[t, 0] = R_batch[t] + td_batch[t] + GAMMA * \
            clipped_rhos[t] * (V_batch[t+1] - R_batch[t+1])
        # td_batch[t]
    # for i in range(len(td_batch)):
    #     a = np.argmax(a_batch[i])
    #     td_batch[i] *= np.clip(probe_batch[i][a] / mu_a_batch[i][a], 0., 1.)

    actor.train_v2(
        s_batch, a_batch, td_batch, mu_a_batch)
    critic.train_v2(s_batch, V_batch)
    return td_batch

def compute_gradients(s_batch, a_batch, r_batch, terminal, actor, critic, mu_a_batch=None):
    """
    batch of s, a, r is from samples in a sequence
    the format is in np.array([batch_size, s/a/r_dim])
    terminal is True when sequence ends as a terminal state
    """
    assert s_batch.shape[0] == a_batch.shape[0]
    assert s_batch.shape[0] == r_batch.shape[0]
    ba_size = s_batch.shape[0]

    v_batch = critic.predict(s_batch)
    probe_batch = actor.predict(s_batch)
    R_batch = np.zeros(r_batch.shape)
    V_batch = np.zeros(r_batch.shape)
    # impala
    clipped_rhos = []
    for i in range(ba_size):
        a = np.argmax(a_batch[i])
        clipped_rhos.append(
            np.clip(probe_batch[i][a] / mu_a_batch[i][a], 0., 1.))

    # clipped_rhos = tf.minimum(clip_rho_threshold, rhos, name='clipped_rhos')
    if terminal:
        R_batch[-1, 0] = 0  # terminal state
    else:
        R_batch[-1, 0] = v_batch[-1, 0]  # boot strap from last state

    for t in reversed(range(ba_size - 1)):
        R_batch[t, 0] = r_batch[t] + GAMMA * R_batch[t + 1, 0]

    td_batch = R_batch - v_batch
    # impala
    for t in reversed(range(ba_size - 1)):
        V_batch[t, 0] = R_batch[t] + td_batch[t] + GAMMA * \
            clipped_rhos[t] * (V_batch[t+1] - R_batch[t+1])
        # td_batch[t]
    # for i in range(len(td_batch)):
    #     a = np.argmax(a_batch[i])
    #     td_batch[i] *= np.clip(probe_batch[i][a] / mu_a_batch[i][a], 0., 1.)

    actor_gradients = actor.get_gradients(
        s_batch, a_batch, td_batch, mu_a_batch)
    critic_gradients = critic.get_gradients(s_batch, V_batch)

    return actor_gradients, critic_gradients, td_batch


def discount(x, gamma):
    """
    Given vector x, computes a vector y such that
    y[i] = x[i] + gamma * x[i+1] + gamma^2 x[i+2] + ...
    """
    out = np.zeros(len(x))
    out[-1] = x[-1]
    for i in reversed(range(len(x)-1)):
        out[i] = x[i] + gamma*out[i+1]
    assert x.ndim >= 1
    # More efficient version:
    # scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]
    return out


def compute_entropy(x):
    """
    Given vector x, computes the entropy
    H(x) = - sum( p * log(p))
    """
    H = 0.0
    for i in range(len(x)):
        if 0 < x[i] < 1:
            H -= x[i] * np.log(x[i])
    return H


def build_summaries():
    td_loss = tf.Variable(0.)
    tf.summary.scalar("TD_loss", td_loss)
    eps_total_reward = tf.Variable(0.)
    tf.summary.scalar("Eps_total_reward", eps_total_reward)

    summary_vars = [td_loss, eps_total_reward]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars
