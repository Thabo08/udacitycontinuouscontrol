
[single]: single_agent_cpu.png "single_agent"
[multi]: multi_agent_cpu.png "multi_agent"

# Deep Deterministic Policy Gradient: Continuous Control Agent
In this project, a double-jointed arm is trained to continuously move to target locations. The agent is trained using the
actor-critic approach, implementing the Deep Deterministic Policy Gradient (DDPG) algorithm.

The environment used, provided by Unity, has a state space of 33 variables, and an action space of 4 variables. The state
space variables correspond to the arm's position, velocity, rotation and angular velocities. The action space 
corresponds to the torque that must be applied to the arm's joints.

The DDPG algorithm, first introduced in https://arxiv.org/pdf/1509.02971.pdf, can learn policies in high 
dimensional continuous, action spaces. The DDPG algorithm is then a prime candidate to solve this continuous control problem
because of the high dimensionality, and the continuous nature of the action space.

The DDPG uses the Actor-Critic approach, meaning there are two models used in the training. The **actor model** is a 
policy function which *deterministically* maps states to action. The **critic model**, is a value based function, "learned
using the Bellman equation like in Q-learning".

The DDPG has the following key features:
* It learns from mini batches, sampled randomly from a replay buffer containing past experiences. The replay buffer allows
for learning from uncorrelated experiences
  
* In training the actor and critic models, target networks of those models are created. The target networks are then taken
through 'soft' update processes, where the network weights are slowly updated. The updates help the target networks track
  the learned networks, improving overall learning stability. The soft updates are driven by a small factor, tau.
  
* Sampled noise is added to the actor policy. This introduces exploration in the training, which is otherwise hard to achieve
in continuous action space problems
  

### Models
The actor model has two hidden layers, with 256 and 128 units respectively. This model takes in a state, and using the 
*tanh* activation function, returns an action represented as a value between -1 and 1 - ideal for continuous action spaces.

The critic model also has two hidden layers, also with 256 and 128 units respectively. The critic takes in a state and an
action and then maps that to a Q value, using the *relu* activation function.

The hyper-parameters used in training the models are given below. The *Adam* optimizer was used for back-propagation in
both models.

Parameter      | Value  | Context | Description |
-------------- | ------ | ------- | ----------- |
Actor learning rate | 1e-4 | Actor | The learning rate of the actor model
Critic learning rate | 1e-4 | Critic | The learning rate of the critic model
Weight decay | 0 | Critic | The weight decay applied in the model when optimizing
Buffer size | 1e6 | Replay Buffer | The size of the replay memory/buffer
Batch size | 128 | Replay Buffer | The mini batch size controlling the sample size from the buffer
Gamma | 0.99 | Critic | The discount factor in the Q-learning
Tau | 1e-3 | Actor/Critic | The value used to make soft updates to the learned networks

For noise sampling, the Ornstein-Uhlenbeck process was employed. This adds temporally correlated noise to the predicted
actions.

### The Agent
Using the actor-critic approach an agent is initialized to learn from the environment by interaction. With interactions 
stored in the replay buffer at a configurable frequency, the agent samples from those experiences and uses those to learn.

The agent has the following high level behaviour:
* Predict actions based on the current state
* Collect experiences at regular intervals
* Learn from collected experiences sampled from the replay buffer. Herein, the following happens:
    * The actions and Q targets are computed from the target networks
    * The expected Q targets are computed from the learned critic network
    * The Q targets from the target network, and the learned network are used to compute the loss and therefore minimize it
    * The learned actor network is used to compute actions based on the state
    * The loss in the actor network is then computed using the predicted actions, and the learned critic network
    * The actor loss is then used to optimize the network weights
    * Both the learned actor and critic networks are then 'softly' updated
  
### Training
The agent is trained in an episodic manner by doing the following per episode, until a target score is reached:
1. Resets the environment states, and the agent noise
1. Takes a number of steps 'through' the environment and does the following:
  1. Takes an action based on the state (invokes the actor model to do this)
  1. Collects an experience tuple from the taken action
  1. Adds the collected experience to the memory buffer and performs the learning if enough samples were collected
  1. Sets the current state to the state returned following the taken action   
  1. Collects a reward based on the action taken (action returned by the actor model)
  1. Completes an episode if a terminal state was reached
1. Evaluates the average score collected; continues if score is not reached, and completes the training if it was reached

### Results
In this project, two versions of the agent were trained and tested. One is a single agent version, and the other is a multi
agent version, which created 20 copies of the agents in the environment, all to be trained.

The single agent trained quicker because there's only one agent to train. The multi agent version, on the other hand trained
over a longer period because of the many agents. However, the multi agent seems to be learning better because of the large
and varied pool of experiences to sample from.

The training results for both versions are given below. Collecting a score of 0.1 for every correct action at every time
step, the target to reach was +30.0.

Version      | Target Score  | Episodes to target | Average Score | Training Time |
------------ | ------------- | ------------------ | ------------- | ------------- |
Single | 30.0 | 371 | 30.04 | ~ 1 hr
Multi | 30.0 | 97 | 30.08 | ~ 3 hr

Below, are the plots of the rewards gained per episode as the agent learns. First shown are for the single agent, the multi agent version.

![single_agent][single]

![multi_agent][multi]


### Ideas of Future Work
Although the DDPG implementation worked well, it's worth a while to explore the following ideas for improving the learning:
* Use Parameter Noise for exploration - This has been shown in some cases to enable quicker training, which can help in pushing for higher agent scores
* Trying other policies like Proximal Policy Optimization and Trust Region Policy Optimization
* Exploring with different hyper-parameters
