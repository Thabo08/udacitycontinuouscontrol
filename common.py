""" Common functionality and modules used by other modules """

import gym
import math
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
