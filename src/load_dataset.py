'''
Author: Jikun Kang
Date: 1969-12-31 19:00:00
LastEditTime: 2023-05-12 15:44:29
LastEditors: Jikun Kang
FilePath: /Hyper-DT/src/load_dataset.py
'''
import argparse
import os
import random
import numpy as np
import h5py
import torch
import pickle


def append_data(data, s, a, r, done, info):
    data['observations'].append(s)
    data['actions'].append(a)
    data['rewards'].append(r)
    data['terminals'].append(done)
    data['infos'].append(info)


def npify(data):
    for k in data:
        if k == 'terminals':
            dtype = np.bool_
        else:
            dtype = np.float32

        data[k] = np.array(data[k], dtype=dtype)

def prepare_dataset2(
        game_name='bigfish',
        mode='toy',
        steps = 1,
):
    # Load dataset
    dataset_dir = f'data/{game_name}/'
    num_files = len([name for name in os.listdir(dataset_dir)
                    if os.path.isfile(os.path.join(dataset_dir, name))])
    all_traj = []

    for i in range(1400, num_files, steps):
        name = f'{game_name}_{i}.hdf5'
        print(f'Reading file {name}')
        file_name = os.path.join(dataset_dir, name)
        with h5py.File(file_name, 'r') as f:
            states = np.array(f.get('observations')).reshape(-1, 3, 64, 64)
            actions = np.array(f.get('actions')).reshape(-1, 1)  # (256, 64)
            rewards = np.array(f.get('rewards')).reshape(-1, 1)
            dones = np.array(f.get('terminals')).reshape(-1, 1)
            all_traj.append({
                'observations': states,
                'actions': actions,
                'rewards': rewards,
                'terminals': dones
            })
    fname = f'data/{game_name}_{mode}.pkl'
    with open(fname, 'wb') as f:
        pickle.dump(all_traj, f)
    print(f"==============>Finished {fname}")

def prepare_dataset(
        folder_name='bigfish',
        game_name='bigfish',
        mode='toy',
        steps = 1000,
):
    # Load dataset
    dataset_dir = f'{folder_name}/{game_name}'
    file_name_list = [name for name in os.listdir(dataset_dir)
                    if os.path.isfile(os.path.join(dataset_dir, name))]
    num_files = len(file_name_list)
    all_traj = []
    if num_files == 0:
        return False

    for name in file_name_list:
        # name = f'{i}.hdf5'
        print(f'Reading file {name}')
        file_name = os.path.join(dataset_dir, name)
        with h5py.File(file_name, 'r') as f:
            states = np.array(f.get('observations')).reshape(-1, 39)
            actions = np.array(f.get('actions')).reshape(-1, 4)  # (256, 4)
            rewards = np.array(f.get('rewards')).reshape(-1, 1)
            dones = np.array(f.get('terminals')).reshape(-1, 1)
            all_traj.append({
                'observations': states,
                'actions': actions,
                'rewards': rewards,
                'terminals': dones
            })
    fname = f'data_success/{game_name}_{mode}.pkl'
    with open(fname, 'wb') as f:
        pickle.dump(all_traj, f)
    print(f"==============>Finished {fname}")


def load_dataset(
    game_name,
    mode='full'
):
    dataset_path = f'data_success/{game_name}_{mode}.pkl'
    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)
    print(f"=========> Loaded dataset {dataset_path}")

    # save all path information into separate lists
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(
        states, axis=0), np.std(states, axis=0) + 1e-6

    num_timesteps = sum(traj_lens)

    print('=' * 50)
    print(f'Starting new experiment: {game_name}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(
        f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)

    return trajectories, returns, traj_lens, states, state_mean, state_std


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t]+gamma*discount_cumsum[t+1]
    return discount_cumsum


def get_batch(
        num_trajectories,
        p_sample,
        trajectories,
        sorted_inds,
        state_dim,
        act_dim,
        state_mean,
        state_std,
        max_len=1,
        device='cpu',
        batch_size=256
):

    batch_inds = np.random.choice(
        np.arange(num_trajectories),
        size=batch_size,
        replace=True,
        p=p_sample
    )

    s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
    for i in range(batch_size):
        traj = trajectories[int(sorted_inds[batch_inds[i]])]
        si = random.randint(0, traj['rewards'].shape[0] - 1)

        # get sequences from dataset
        s.append(traj['observations']
                 [si:si + max_len].reshape(1, -1, state_dim))
        a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
        r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
        if 'terminals' in traj:
            d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
        else:
            d.append(traj['dones'][si:si + max_len].reshape(1, -1))
        # timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
        # timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len - \
            1  # padding cutoff
        # rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[
        #            :s[-1].shape[1] + 1].reshape(1, -1, 1))
        # if rtg[-1].shape[1] <= s[-1].shape[1]:
        #     rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

        # padding and state + reward normalization
        tlen = s[-1].shape[1]
        s[-1] = np.concatenate([np.zeros((1, max_len -
                               tlen, state_dim)), s[-1]], axis=1)
        s[-1] = (s[-1] - state_mean) / state_std
        a[-1] = np.concatenate([np.ones((1, max_len -
                               tlen, act_dim)) * -10., a[-1]], axis=1)
        r[-1] = np.concatenate([np.zeros((1, max_len -
                               tlen, 1)), r[-1]], axis=1)
        d[-1] = np.concatenate([np.ones((1, max_len - tlen))
                               * 2, d[-1]], axis=1)
        # rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)),
        #                          rtg[-1]], axis=1) / scale
        # timesteps[-1] = np.concatenate([np.zeros((1,
        #                                max_len - tlen)), timesteps[-1]], axis=1)
        # mask.append(np.concatenate(
        #     [np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

    s = torch.from_numpy(np.concatenate(s, axis=0)).to(
        dtype=torch.float32, device=device)
    a = torch.from_numpy(np.concatenate(a, axis=0)).to(
        dtype=torch.float32, device=device)
    r = torch.from_numpy(np.concatenate(r, axis=0)).to(
        dtype=torch.float32, device=device)
    d = torch.from_numpy(np.concatenate(d, axis=0)).to(
        dtype=torch.long, device=device)
    # rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(
    #     dtype=torch.float32, device=device)
    # timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(
    #     dtype=torch.long, device=device)
    # mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

    return s, a, r, d 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder_name", default="dataset_success", type=str)
    parser.add_argument("--mode", default='full', type=str)
    parser.add_argument("--steps", default=1, type=int)
    args = parser.parse_args()
    meta_world_game_list = ['assembly-v2', 'basketball-v2', 'button-press-topdown-v2', 
    'button-press-topdown-wall-v2', 'button-press-v2', 'button-press-wall-v2', 'coffee-button-v2', 'coffee-pull-v2', 'coffee-push-v2', 'dial-turn-v2', 'disassemble-v2', 'door-close-v2', 'door-open-v2', 'drawer-close-v2', 'drawer-open-v2', 'faucet-open-v2', 'faucet-close-v2', 'hammer-v2', 'handle-press-side-v2', 'handle-press-v2', 'handle-pull-side-v2', 'handle-pull-v2', 'lever-pull-v2', 'peg-insert-side-v2', 'pick-place-wall-v2', 'pick-out-of-hole-v2', 'reach-v2', 'push-back-v2', 'push-v2', 'pick-place-v2', 'plate-slide-v2', 'plate-slide-side-v2', 'plate-slide-back-v2', 'plate-slide-back-side-v2', 'peg-unplug-side-v2', 'soccer-v2', 'stick-push-v2', 'stick-pull-v2', 'push-wall-v2', 'reach-wall-v2', 'shelf-place-v2', 'sweep-into-v2', 'sweep-v2', 'window-open-v2', 'window-close-v2']
    for name in meta_world_game_list:
        prepare_dataset(folder_name=args.folder_name, game_name=name, mode=args.mode, steps=args.steps)
    # load_dataset('bigfish')
    # prepare_dataset2(mode=args.mode)
