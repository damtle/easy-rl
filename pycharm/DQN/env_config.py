import random
import gym
import os
import numpy as np
import torch
from MLP import MLP
from ReplayBuffer import ReplayBuffer
from DQN import DQN


def all_seed(env, seed = 1):
    env.reset(seed=seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def env_agent_config(cfg):
    env = gym.make(cfg['env_name'], new_step_api=True)
    if cfg['seed'] != 0:
        all_seed(env, seed=cfg['seed'])
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    print(f"state space dimension: {n_states}, action space dimension: {n_actions}")
    cfg.update({"n_states":n_states, "n_actions":n_actions})
    model = MLP(n_states, n_actions, hidden_dim=cfg['hidden_dim'])
    memory = ReplayBuffer(cfg['memory_capacity'])
    agent = DQN(model, memory, cfg)
    return env, agent
