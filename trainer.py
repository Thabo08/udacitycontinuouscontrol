""" Train the agent """

from collections import deque
import argparse
import datetime
from unityagents import UnityEnvironment

from agent import ContinuousControlAgent
from common import *
from support import Experience, ReplayBuffer
import config


def environment_settings(file_name):
    settings = {}
    env = UnityEnvironment(file_name=file_name)
    settings["env"] = env

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    settings["brain_name"] = brain_name

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents in the environment
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)
    settings["num_agents"] = num_agents

    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)
    settings["action_size"] = action_size

    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('States look like:', states[0])
    print('States have length:', state_size)
    settings["state_size"] = state_size

    return settings


def step_tuple(env_info, single_agent):
    """ Returns a tuple of next state, reward, and done when the agent steps through the environment based
        on the action taken
        :param env_info: Object holding information about the environment at a certain point
        :param single_agent: Indicates whether to load a single or a multi agent environment with all settings needed
    """
    if single_agent:
        return env_info.vector_observations[0], env_info.rewards[0], env_info.local_done[0]
    return env_info.vector_observations, env_info.rewards, env_info.local_done


def plot(stats):
    scores = stats["scores"]
    episodes = stats["episodes"]
    plt.ylabel("Score")
    plt.xlabel("Episode #")
    plt.plot(episodes, scores)
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    plt.show()


def ddpg(agent: ContinuousControlAgent, env_settings: dict, single_agent, num_episodes=2000, target=30.,
         max_time_steps=500, saved_model="checkpoint.pth"):
    """ Train an agent using the DDPG algorithm

        :param single_agent: Indicates whether to load a single or a multi agent environment with all settings needed
        :param env_settings: Settings of the environment
        :param agent: a continuous control agent
        :param num_episodes: the number of episodes to train the agent
        :param target: The average target score the agent needs to achieve for optimal performance
        :param max_time_steps: Maximum time steps per episode
        :param saved_model: The file path to save the model weights
    """
    now = datetime.datetime.now()
    print(now, "- Training {}".format("single agent" if single_agent else "{} agents".format(20))
          + " for max {} episodes. Target score to reach is {}".format(num_episodes, target))
    # collections to help keep track of the score
    scores_deque = deque(maxlen=100)
    scores = []
    stats = {"scores": [], "episodes": []}  # collects stats for plotting purposes
    mean_score = 0.

    env = env_settings["env"]
    brain_name = env_settings["brain_name"]
    num_agents = env_settings["num_agents"]

    for episode in range(1, num_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        states = env_info.vector_observations[0] if single_agent else env_info.vector_observations
        agent.reset()  # reset the agent noise
        score = 0 if single_agent else np.zeros(num_agents)

        for _ in range(max_time_steps):
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]
            next_states, rewards, dones = step_tuple(env_info, single_agent)
            if single_agent:
                agent.step(Experience(states, actions, rewards, next_states, dones))
            else:
                # randomly select 10 agents to train
                for idx in random.sample(range(num_agents), 10):
                    agent.step(Experience(states[idx], actions[idx], rewards[idx], next_states[idx], dones[idx]))
            states = next_states
            score += rewards
            if np.any(dones):
                break
        scores_deque.append(score)
        scores.append(score)
        mean_score = np.mean(scores_deque)

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, mean_score), end="")
        if episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, mean_score))

        stats["scores"].append(score if single_agent else np.mean(score))
        stats["episodes"].append(episode)

        if mean_score >= target:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode, mean_score))
            print("Target score of {0} has been reached. Saving model to {1}".format(target, saved_model))
            torch.save(agent.local_actor_network().state_dict(), saved_model)
            break

    now = datetime.datetime.now()
    print(now, "- Finished training " + "successfully!" if mean_score >= target else "unsuccessfully!")
    return scores, stats


def run(agent_: ContinuousControlAgent, env_settings: dict, single_agent: bool, num_episodes=2000,
        max_time_steps=1000, target=30., saved_model="checkpoint.pth"):
    try:
        _, stats = ddpg(agent_, env_settings, single_agent, num_episodes=num_episodes, target=target,
                        max_time_steps=max_time_steps,
                        saved_model=saved_model)
        plot(stats)
    finally:
        # make sure the environment gets closed regardless of what happens
        env_settings["env"].close()


def test(agent_: ContinuousControlAgent, env_settings, single_agent, filename):
    print("Loading weights from {} to test the agent".format(filename))
    agent_.local_actor_network().load_state_dict(torch.load(filename))
    env = env_settings["env"]
    brain_name = env_settings["brain_name"]
    num_agents = env_settings["num_agents"]

    env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
    state = env_info.vector_observations[0] if single_agent else env_info.vector_observations  # get the current state
    score = 0 if single_agent else np.zeros(num_agents)  # initialize the score
    while True:
        action = agent_.act(state, add_noise=False)  # select an action
        env_info = env.step(action)[brain_name]  # send the action to the environment
        next_state = env_info.vector_observations[0] if single_agent else env_info.vector_observations
        # get the next state
        reward = env_info.rewards[0] if single_agent else env_info.rewards  # get the reward
        done = env_info.local_done[0] if single_agent else env_info.local_done  # see if episode has finished
        score += reward  # update the score
        state = next_state  # roll over the state to next time step
        if np.any(done):  # exit loop if episode finished
            break

    score_ = lambda x: np.round(x, 2)
    print("Score for {} agent(s): {}".format(num_agents, score_(score) if single_agent else score_(np.mean(score))))

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""This script trains a continuous control agent using the 
    Deep Deterministic Policy Gradient (DDPG) algorithm""")
    parser.add_argument("agents", choices=["single", "multi"], help="Number of agents to train")
    parser.add_argument("agentFile", help="The file to load the agent(s)")
    parser.add_argument("--model", default="checkpoint.pth", help="Path where the trained model should be saved")
    parser.add_argument("--updateFrequency", default=2, help="Frequency with which to update models")
    parser.add_argument("--mode", default="train", choices=["train", "test"],
                        help="Mode describing whether to train or test")

    args = parser.parse_args()

    single_agent = args.agents == "single"
    env_settings = environment_settings(args.agentFile)
    num_agents = env_settings["num_agents"]

    if single_agent and num_agents > 1:
        raise RuntimeError("Loaded incorrect single agent file")
    if not single_agent and num_agents == 1:
        raise RuntimeError("Loaded incorrect multi agent file")

    action_size = env_settings["action_size"]
    state_size = env_settings["state_size"]
    memory = ReplayBuffer(action_size, config.BUFFER_SIZE, config.BATCH_SIZE, random_seed=0)
    agent = ContinuousControlAgent(state_size, action_size, 0, memory=memory, update_frequency=args.updateFrequency)
    train = args.mode == "train"
    if train:
        run(agent, env_settings, single_agent, num_episodes=100, max_time_steps=500, target=30., saved_model=args.model)
    else:
        test(agent, env_settings, single_agent, filename=args.model)
