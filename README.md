
[multi]: multi_agent_cpu.png "multi_agent"
[env]: single_env.png "env"

![multi_agent][multi]

# Deep Deterministic Policy Gradient: Continuous Control Agent

In this project, single and multi agents of double-jointed arms were trained to continuously move to target locations.
The algorithm used is the Deep Deterministic Policy Gradient (DDPG), which uses two models:
1. **Actor Model**: A policy function based model used to chose actions to take given a state in the environment,
1. **Critic Model**: A value function based model used to critic the chosen actions

This approach is commonly referred to as the Actor-Critic method.

### 1. The Environment
The Unity environment used here is such that there is one, or many double-jointed arms (see figure below). The observation
space has 33 variables, corresponding to the arm's position, velocity, rotation and angular velocities. The action space
4 variables corresponding to the torque that must be applied to the arm's joints.

![env][env]

The goal of the agent is for the arm to continuously track a target (turquoise ball in the figure) by applying torque to
the arm. For every time step this happens, the agent receives a reward of +0.1. An agent is considered to be well-trained
if the agent can accumulate a score of +30 (adding rewards gained) over an average of 100 consecutive episodes.