""" Train the agent """

from agent import ContinuousControlAgent


def ddpg(agent: ContinuousControlAgent, env, num_episodes=2000):
    """ Train an agent using the DDPG algorithm

        :param agent: a continuous control agent
        :param env: environment the agent interacts with
        :param num_episodes: the number of episodes to train the agent
    """
