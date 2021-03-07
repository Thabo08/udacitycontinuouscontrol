""" Train the agent """

from agent import ContinuousControlAgent
from unityagents import UnityEnvironment
from common import *
from collections import deque
from support import Experience

# todo: Make this a system arg
file_name = "/Users/thabo/Documents/code/personal/courses/Udacity/Reinforcement Learning/deep-reinforcement-learning/" \
            "p2_continuous-control/Reacher 2.app"
env = UnityEnvironment(file_name=file_name)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('States look like:', states)
print('States have length:', state_size)


def step_tuple(env_info):
    """ Returns a tuple of next state, reward, and done when the agent steps through the environment based
        on the action taken
        :param env_info: Object holding information about the environment at a certain point
    """
    next_state, reward, done = env_info.vector_observations[0], env_info.rewards[0], env_info.local_done[0]
    return next_state, reward, done


def plot(stats):
    scores = stats["scores"]
    episodes = stats["episodes"]
    plt.ylabel("Score")
    plt.xlabel("Episode #")
    plt.plot(episodes, scores)
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    plt.show()


def ddpg(agent: ContinuousControlAgent, env: UnityEnvironment, num_episodes=2000, max_time_steps=1000, target=30.,
         saved_model="checkpoint.pth"):
    """ Train an agent using the DDPG algorithm

        :param agent: a continuous control agent
        :param env: environment the agent interacts with
        :param num_episodes: the number of episodes to train the agent
        :param max_time_steps: Maximum number of time steps to interact with the environment per episode
        :param target: The average target score the agent needs to achieve for optimal performance
        :param saved_model: The file path to save the model weights
    """
    print("Training agent for max {} episodes. Target score to reach is {}".format(num_episodes, target))
    # collections to help keep track of the score
    scores_deque = deque(maxlen=100)
    scores = []
    stats = {"scores": [], "episodes": []}  # collects stats for plotting purposes
    mean_score = 0.

    for episode in range(1, num_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        state = env_info.vector_observations[0]
        score = 0
        for step in range(max_time_steps):
            action = agent.act(state)
            env_info = env.step(action)[brain_name]
            next_state, reward, done = step_tuple(env_info)
            agent.step(Experience(state, action, reward, next_state, done))
            state = next_state
            score += reward
            if done:
                break
        scores_deque.append(score)
        scores.append(score)
        mean_score = np.mean(scores_deque)

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, mean_score), end="")
        if episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, mean_score))

        stats["scores"].append(score)
        stats["episodes"].append(episode)

        if mean_score >= target:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode, mean_score))
            print("Target score of {0} has been reached. Saving model to {1}".format(target, saved_model))
            torch.save(agent.local_actor_network().state_dict(), saved_model)
            break

    print("Finished training " + "successfully!" if mean_score >= target else "unsuccessfully!")
    return scores, stats


def run(agent_: ContinuousControlAgent, env_: UnityEnvironment, num_episodes=2000, max_time_steps=1000, target=30.,
        saved_model="checkpoint.pth"):
    try:
        _, stats = ddpg(agent_, env_, num_episodes=num_episodes, max_time_steps=max_time_steps, target=target,
                        saved_model=saved_model)
        plot(stats)
    finally:
        # make sure the environment gets closed regardless of what happens
        env_.close()


def test(agent_: ContinuousControlAgent, filename):
    agent_.local_actor_network().load_state_dict(torch.load(filename))

    env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
    state = env_info.vector_observations[0]  # get the current state
    score = 0  # initialize the score
    while True:
        action = agent_.act(state, .0)  # select an action
        env_info = env.step(action)[brain_name]  # send the action to the environment
        next_state = env_info.vector_observations[0]  # get the next state
        reward = env_info.rewards[0]  # get the reward
        done = env_info.local_done[0]  # see if episode has finished
        score += reward  # update the score
        state = next_state  # roll over the state to next time step
        if done:  # exit loop if episode finished
            break

    print("Score: {}".format(score))

    env.close()


if __name__ == '__main__':
    adaptive_noise = False
    # agent = ContinuousControlAgent(state_size, action_size, random_seed=0, adaptive_noise=adaptive_noise,
    #                                update_frequency=4)
    # filename = "checkpoint-adaptive.pth" if adaptive_noise else "checkpoint-ornsteinuhlenbeck.pth"
    # train_mode = True
    # if train_mode:
    #     run(agent, env, num_episodes=100, max_time_steps=500, target=1., saved_model=filename)
    # else:
    #     test(agent, filename)
