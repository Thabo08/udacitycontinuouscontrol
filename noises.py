""" Action and Policy Parameter noise implementations """

from common import *
import copy


class ActionNoise:
    def reset(self):
        pass


class OUNoise(ActionNoise):
    """Ornstein-Uhlenbeck exploration noise process for temporally correlated noise"""

    def __init__(self, action_size, seed, mu=0.0, theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(action_size)
        self.theta = theta
        self.sigma = sigma
        self.state = None
        self.action_size = action_size
        random.seed(seed)

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(
            self.action_size
        )
        self.state = x + dx
        return self.state


class AdaptiveNoise(ActionNoise):
    """Adds adaptive noise to the parameters of the neural network policy - allows for quicker training
    For more details, see: https://openai.com/blog/better-exploration-with-parameter-noise/
    """

    def __init__(
        self, initial_stddev=0.1, desired_action_stddev=0.1, adoption_coefficient=1.01
    ):
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adoption_coefficient = adoption_coefficient
        self.current_stddev = initial_stddev

    def adapt(self, distance):
        if distance > self.desired_action_stddev:
            self.current_stddev /= (
                self.adoption_coefficient
            )  # decrease standard deviation
        else:
            self.current_stddev *= (
                self.adoption_coefficient
            )  # increase standard deviation
