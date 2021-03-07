""" Action and Policy Parameter noise implementations """

from common import *
import copy


class ActionNoise:
    def reset(self):
        pass


# class OUNoise(ActionNoise):
#     """ Ornstein-Uhlenbeck exploration noise process for temporally correlated noise """
#
#     def __init__(self, action_size, seed, mu=0., theta=.15, sigma=.2):
#         self.mu = mu * np.ones(action_size)
#         self.theta = theta
#         self.sigma = sigma
#         self.state = None
#         random.seed(seed)
#
#     def reset(self):
#         """Reset the internal state (= noise) to mean (mu)."""
#         self.state = copy.copy(self.mu)
#
#     def sample(self):
#         """Update internal state and return it as a noise sample."""
#         # x = self.state
#         # dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for _ in range(len(x))])
#         # self.state = x + dx
#         return self.state

class OUNoise(ActionNoise):
    def __init__(self, mu, sigma=.2, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.x_prev = None
        self.reset()

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random\
            .normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class AdaptiveNoise(ActionNoise):
    """ Adds adaptive noise to the parameters of the neural network policy - allows for quicker training
        For more details, see: https://openai.com/blog/better-exploration-with-parameter-noise/
     """
    def __init__(self, initial_stddev=.1, desired_action_stddev=.1, adoption_coefficient=1.01):
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adoption_coefficient = adoption_coefficient
        self.current_stddev = initial_stddev

    def adapt(self, distance):
        if distance > self.desired_action_stddev:
            self.current_stddev /= self.adoption_coefficient  # decrease standard deviation
        else:
            self.current_stddev *= self.adoption_coefficient  # increase standard deviation
