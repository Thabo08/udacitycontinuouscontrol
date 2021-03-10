
[single]: single_agent_cpu.png "single_agent"
[multi]: multi_agent_cpu.png "multi_agent"
[env]: single_env.png "env"
[single_mov]: single_agent.gif "single_mov"
[multi_mov]: multi_agent.gif "multi_mov"


# Deep Deterministic Policy Gradient: Continuous Control Agent

In this project, single and multi agents of double-jointed arms were trained to continuously move to target locations.
The algorithm used is the Deep Deterministic Policy Gradient (DDPG), which uses two models:
1. **Actor Model**: A policy function based model used to chose actions to take given a state in the environment,
1. **Critic Model**: A value function based model used to critic the chosen actions

This approach is commonly referred to as the Actor-Critic method.

### 1. The Environment
The Reacher (Unity) environment used here is such that there is one, or many double-jointed arms (see figure below). The observation
space has 33 variables, corresponding to the arm's position, velocity, rotation and angular velocities. The action space
4 variables corresponding to the torque that must be applied to the arm's joints.

![env][env]

The goal of the agent is for the arm to continuously track a target (turquoise ball in the figure) by applying torque to
the arm. For every time step this happens, the agent receives a reward of +0.1. An agent is considered to be well-trained
if the agent can accumulate a score of +30 (adding rewards gained) over an average of 100 consecutive episodes.

### 2. The Model
As mentioned, the agent uses the Actor-Critic approach method of learning, where 2 models are used, one as an actor and 
the other as the actor's critic. The Actor model takes the current state as input and outputs a suitable action. The 
critic takes both a state, and the action returned by the actor, as input and returns Q values (hence value based).

The architectures of these models are defined in the _models.py_ file. The actor model has 2 hidden layers and outputs 
and using the *tanh* activation function outputs a value between -1 and 1, representing an action. The critic model also
has 2 hidden layers and uses a *relu* function to map states and action to Q values.

### 3. The Agent
The agent, which implements the DDPG algorithm, can be found in the _agent.py_ file. The agent initialises and uses the 
following key components:
1. **Actor Model**: See above
1. **Critic Model**: See above
1. **Replay Buffer**: A container used to keep experiences from the agent's interaction with the environment. This buffer is used for sampling experiences in order to avoid correlations between states.
1. **Noise Object**: Implemented here using the **Ornstein-Uhlenbeck** algorithm, the noise object is used in introducing exploration by adding temporally correlated noise to the actions 

The agent has the following high level functionalities, which enable it to learn from interactions:
1. **Noise reset**: At the beginning of every episode, the agent resets the noise to start each episode's noise factor on a clean slate
1. **Act**: The agent uses this to take an action based the input state
1. **Step and Learn**: The agent uses this to 'step' into the environment and then learn from experiences sampled from the replay buffer

### 4. The Trainer
The code to train the agent can be found in _trainer.py_ file. This is where a Unity environment is loaded, which then 
provides the environment to use with training the agent.

Herein, an agent is trained over the desired number of episodes, with the target score also optionally specified. The 
implementation in the trainer is such that a single agent, or a multi (20) agent environment can be loaded and trained.
Additionally, there's logic to test the trained agent.

### 5. Usage
The packages required to run the _trainer.py_ script can be installed this way:
```
pip install -r requirements.txt
```
The links to download the environment for popular operating systems per version of the environment are as follows:

Version      |    Mac OSX    |   Linux   | Windows 32-bit | Windows 64-bit |
------------ | ------------- | --------- | -------------- | -------------- |
Single (1 agent) | [Click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip) | [Click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip) | [Click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip) | [Click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)
Multi (20 agents)| [Click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip) | [Click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip) | [Click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip) | [Click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

Once downloaded, place the zip archive in the directory of your choice and decompress the file.

The script is command line based and can be used in the following ways:
```commandline
python trainer.py -h
```
The line above gives a help print out of the high level parameters that can be passed to run the trainer
```
python trainer.py <agents> <environment_file> --mode <mode> --model <saved_model>
```
The line above shows how to run the agent, where ```<agents>``` indicates how many agents to used (the options are single or multi),
```<environment_file>``` is used to specify which environment file to load, guided by the choice of agents, ```<mode>```
indicates which mode to run the trainer in, with options being trained or tested, lastly, ```<saved_model>``` specifies where
the trained model weights must be saved when training and where they must be loaded from when testing.

See below as examples of running the script for single agent in train mode, and the multi agent in test mode:
```commandline
python trainer.py single Reacher.app --mode train --model checkpoint_single.pth
python trainer.py multi Reacher.app --mode test --model checkpoint_multi.pth
```

### 6. Results
The trainer was used to train both single and multi versions of the environment. The results are shown below:

Version      | Target Score  | Episodes to target | Average Score | Training Time |
------------ | ------------- | ------------------ | ------------- | ------------- |
Single | 30.0 | 371 | 30.04 | ~ 1 hr
Multi | 30.0 | 97 | 30.08 | ~ 3 hr

Below, are the plots of the rewards gained per episode as the agent learns. First shown are for the single agent, the multi agent version.

![single_agent][single]

![multi_agent][multi]

Below, are the illustrations of how the trained agents perform in test mode. Shown first is the single agent version, 
followed by the multi agent version

![single_mov][single_mov]

![multi_mov][multi_mov]

The average scores reached by the agent in single and multi modes are **35.59** and **39.47**, respectively. These scores meet the
minimum required score of **30.0** for this setting.

