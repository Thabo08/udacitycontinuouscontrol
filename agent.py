""" Implementation of the continuous control agent. The agent implements the Deep Deterministic Policy Gradient
 (DDPG) algorithm"""
from torch import optim

import config
from common import *
from models import Actor
from models import Critic
from noises import OUNoise
from support import Experience
from support import ReplayBuffer


def minimize_loss(loss, optimizer: optim.Adam, is_critic=False, critic=None):
    optimizer.zero_grad()
    loss.backward()
    if is_critic and critic is not None:
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1)
    optimizer.step()


def soft_update(local_model, target_model):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target

    Params
    ======
        local_model: PyTorch model (weights will be copied from)
        target_model: PyTorch model (weights will be copied to)
        tau (float): interpolation parameter
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(config.TAU * local_param.data + (1.0 - config.TAU) * target_param.data)


class ContinuousControlAgent:
    """ The agent that learns to interact with an environment using the DDPG algorithm """
    def __init__(self, state_size, action_size, random_seed, memory: ReplayBuffer, update_frequency=10):
        """
        Initialise the agent
        :param state_size: Dimension of the state space
        :param action_size: Dimension of the action space
        :param random_seed: Random seed
        """
        random.seed(random_seed)

        self.time_step = 0
        self.update_frequency = update_frequency

        # Initialise the Actor networks (local and target), including the Optimizer
        self.actor_local = Actor("Actor: Local", state_size, action_size, random_seed).to(DEVICE)
        self.actor_target = Actor("Actor: Target", state_size, action_size, random_seed).to(DEVICE)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=config.ACTOR_LR)

        # Initialise the Critic networks (local and target)
        self.critic_local = Critic("Critic: Local", state_size, action_size, random_seed).to(DEVICE)
        self.critic_target = Critic("Critic: Target", state_size, action_size, random_seed).to(DEVICE)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=config.CRITIC_LR,
                                           weight_decay=config.WEIGHT_DECAY)

        # Exploration noise process
        self.noise = OUNoise(action_size, 0)

        # Replay buffer
        self.memory = memory

        self.ready_to_learn = len(self.memory) > config.BATCH_SIZE

    def reset(self):
        self.noise.reset()

    def act(self, state, add_noise=True):
        """ Return the action for the state as per the policy """
        state = torch.from_numpy(state).float().to(DEVICE)
        self.actor_local.eval()  # put the policy in evaluation mode
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()  # put policy back in training mode
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def step(self, experience: Experience):
        """ Add experiences to the experience buffer and learn from a batch """
        self.memory.add(experience)
        if not self.ready_to_learn:
            if len(self.memory) % (config.BATCH_SIZE // 4) == 0:
                print("Agent has not collected enough experiences to start learning. Collected {}, requires at least {}"
                      " experiences".format(len(self.memory), config.BATCH_SIZE))
            self.ready_to_learn = len(self.memory) > config.BATCH_SIZE

        self.time_step = (self.time_step + 1) % self.update_frequency
        if self.time_step == 0:
            if self.ready_to_learn:
                experiences = self.memory.sample()
                self.learn_(experiences)

    def local_actor_network(self):
        return self.actor_local

    def learn_(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        actions_next = self.actor_target(next_states)
        q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i) - Check https://arxiv.org/pdf/1509.02971.pdf paper
        q_targets = rewards + (config.GAMMA * q_targets_next * (1 - dones))
        # compute critic loss
        q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(q_expected, q_targets)
        minimize_loss(critic_loss, self.critic_optimizer, is_critic=True, critic=self.critic_local)

        # update the actor
        actions_predicted = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_predicted).mean()
        minimize_loss(actor_loss, self.actor_optimizer)

        # update target networks
        soft_update(self.critic_local, self.critic_target)
        soft_update(self.actor_local, self.actor_target)


