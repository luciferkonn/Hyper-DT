'''
Author: Jikun Kang
Date: 1969-12-31 19:00:00
LastEditTime: 2023-05-10 17:39:13
LastEditors: Jikun Kang
FilePath: /Hyper-DT/src/create_self_dataset.py
'''
import csv
import logging
# make deterministic
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch.utils.data import Dataset
from collections import deque
import random
import torch
import pickle
import argparse
from src.load_dataset import get_batch, load_dataset
from src.env_utils import LIMITED_ACTION_TO_FULL_ACTION
from src.fixed_replay_buffer import FixedReplayBuffer


def create_self_dataset(
    num_buffers,
    num_steps,
    game_name,
    trajectories_per_buffer
):
    # -- load data from memory (make more efficient)
    obss = []
    actions = []
    returns = [0]
    done_idxs = []
    stepwise_returns = []

    transitions_per_buffer = np.zeros(50, dtype=int)
    num_traj = 0
    trajectories, returns, traj_lens, states, state_mean, state_std = load_dataset(game_name)
    num_timesteps = sum(traj_lens)
    num_timesteps = max(int(1.*num_timesteps), 1)
    sorted_inds = np.argsort(returns)
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])
    while len(obss) < num_steps:
        buffer_num = np.random.choice(np.arange(50 - num_buffers, 50), 1)[0]
        i = transitions_per_buffer[buffer_num]
        print('loading from buffer %d which has %d already loaded' %
              (buffer_num, i))
        done = False
        curr_num_transitions = len(obss)
        trajectories_to_load = trajectories_per_buffer
        while not done:
            states, ac, ret, terminal = get_batch(
                num_trajectories=num_trajectories,
                p_sample=p_sample,
                trajectories=trajectories,
                sorted_inds=sorted_inds,
                state_dim=39,
                act_dim=4,
                state_mean=state_mean,
                state_std=state_std,
                batch_size=1)
            # (1, 84, 84, 4) --> (4, 84, 84)
            # states = states.transpose((0, 3, 1, 2))[0]
            obss += [states]
            actions += [ac]
            stepwise_returns += [ret[0]]
            if terminal[0]:
                done_idxs += [len(obss)]
                returns += [0]
                if trajectories_to_load == 0:
                    done = True
                else:
                    trajectories_to_load -= 1
            returns[-1] += ret[0]
            i += 1
            if i >= 100000:
                obss = obss[:curr_num_transitions]
                actions = actions[:curr_num_transitions]
                stepwise_returns = stepwise_returns[:curr_num_transitions]
                returns[-1] = 0
                i = transitions_per_buffer[buffer_num]
                done = True
        num_traj += (trajectories_per_buffer -
                                trajectories_to_load)
        transitions_per_buffer[buffer_num] = i
        print('this buffer has %d loaded transitions and there are now %d transitions total divided into %d trajectories' % (
            i, len(obss), num_traj))

    actions = torch.stack(actions).squeeze(1).squeeze(1).numpy()
    # returns = stack(returns).squeeze(1)
    stepwise_returns = torch.stack(stepwise_returns).squeeze(1)
    done_idxs = np.array(done_idxs)

    # -- create reward-to-go dataset
    start_index = 0
    rtg = np.zeros_like(stepwise_returns)
    for i in done_idxs:
        i = int(i)
        curr_traj_returns = stepwise_returns[start_index:i]
        for j in range(i-1, start_index-1, -1):  # start from i-1
            rtg_j = curr_traj_returns[j-start_index:i-start_index]
            rtg[j] = sum(rtg_j)
        start_index = i
    print('max rtg is %d' % max(rtg))

    # -- create timestep dataset
    start_index = 0
    timesteps = np.zeros(len(actions)+1, dtype=int)
    for i in done_idxs:
        i = int(i)
        timesteps[start_index:i+1] = np.arange(i+1 - start_index)
        start_index = i+1
    print('max timestep is %d' % max(timesteps))

    # convert to torch.Tensor
    obss = torch.stack(obss).squeeze(1).squeeze(1)
    actions = torch.from_numpy(actions)
    # returns = torch.from_numpy(done_idxs)
    rtg = torch.from_numpy(rtg)
    timesteps = torch.from_numpy(timesteps)
    # stepwise_returns = torch.from_numpy(stepwise_returns)

    return obss, actions, done_idxs, rtg, timesteps, stepwise_returns
