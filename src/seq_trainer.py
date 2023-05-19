import os
import numpy as np
import torch
import time

import wandb

class Trainer:

    def __init__(
            self, 
            model, 
            optimizer, 
            batch_size, 
            get_batch, 
            loss_fn, 
            num_trajectories,
            p_sample,
            trajectories,
            sorted_inds,
            state_mean,
            state_std,
            run_dir,
            eval_envs,
            eval_name_list,
            train_envs,
            train_name_list,
            scheduler=None, 
            device='cpu',
    ):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.diagnostics = dict()
        self.eval_envs = eval_envs
        self.eval_name_list = eval_name_list
        self.train_envs = train_envs
        self.train_name_list = train_name_list
        self.state_dim = 39
        self.act_dim = 4

        self.num_trajectories = num_trajectories
        self.p_sample = p_sample
        self.trajectories = trajectories
        self.sorted_inds = sorted_inds
        self.state_mean = state_mean
        self.state_std = state_std
        self.run_dir = run_dir
        self.device = device

        self.start_time = time.time()

    def train_iteration(self, num_steps, iter_num=0, print_logs=False, save_freq=1000):
        print(f"=====> Start training {self.train_name_list[0]}")

        train_losses = []
        logs = dict()

        train_start = time.time()

        self.model.train()
        tf_file_loc = os.path.join(self.run_dir, f'tf_model.pt')
        for epoch in range(num_steps):
            train_loss = self.train_step()
            train_losses.append(train_loss)
            if self.scheduler is not None:
                self.scheduler.step()
            if epoch % save_freq == 0 or epoch == (num_steps- 1):
                print("========================")
                print(f"The model is saved to {tf_file_loc}")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    # 'loss': logs['train_loss']
                }, tf_file_loc)
                # wandb.save(tf_file_loc)

        logs['time/training'] = time.time() - train_start

        eval_start = time.time()

        self.model.eval()
        outputs = self.eval_fn(self.model, self.train_name_list, self.train_envs)
        for k, v in outputs.items():
            logs[f'evaluation/{k}'] = v

        logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs

    def train_step(self):
        states, actions, rewards, dones, attention_mask, returns = self.get_batch(self.batch_size)
        state_target, action_target, reward_target = torch.clone(states), torch.clone(actions), torch.clone(rewards)

        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, masks=None, attention_mask=attention_mask, target_return=returns,
        )

        # note: currently indexing & masking is not fully correct
        loss = self.loss_fn(
            state_preds, action_preds, reward_preds,
            state_target[:,1:], action_target, reward_target[:,1:],
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()
    
    def eval_fn(self, model, name_list, env_list):
        returns, lengths = [], []
        success_rate_list = []
        target_rew = 1200
        for name, env in zip(name_list, env_list):
            print(f"=======> Start evaluating {name}")
            with torch.no_grad():
                ret, length, success_rate = evaluate_episode_rtg(
                    env,
                    39,
                    4,
                    model,
                    max_ep_len=1000,
                    scale=1000,
                    target_return=1200/1000,
                    mode='normal',
                    state_mean=self.state_mean,
                    state_std=self.state_std,
                    device=self.device,
                )
            returns.append(ret)
            lengths.append(length)
            success_rate_list.append(success_rate)
        return {
            f'{name}_target_{target_rew}_return_mean': np.mean(returns),
            f'{name}_target_{target_rew}_return_std': np.std(returns),
            f'{name}_target_{target_rew}_length_mean': np.mean(lengths),
            f'{name}_target_{target_rew}_length_std': np.std(lengths),
            f'{name}_success_rate': np.mean(np.array(success_rate_list)),
        }
    
    def evaluate(self):
        logs = dict()
        outputs = self.eval_fn(self.model, self.eval_name_list, self.eval_envs)
        for k, v in outputs.items():
            logs[f'evaluation/{k}'] = v
        print('=' * 80)
        for k, v in logs.items():
            print(f'{k}: {v}')
        return logs
    
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer




class SequenceTrainer(Trainer):

    def train_step(self):
        states, actions, rewards, done, rtg, timesteps, attention_mask = self.get_batch(
            self.num_trajectories,
            self.p_sample,
            self.trajectories,
            self.sorted_inds,
            self.state_dim,
            self.act_dim,
            self.state_mean,
            self.state_std,
            batch_size=self.batch_size,
            device=self.device,
            )
        action_target = torch.clone(actions)

        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask,
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        return loss.detach().cpu().item()

def evaluate_episode_rtg(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
    ):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    success = 0
    episode_return_list = []
    episode_length_list = []
    ep_return = target_return
    for i in range(20):
        env.reset()
        env.reset_model()
        state = env.reset()
        if mode == 'noise':
            state = state + np.random.normal(0, 0.1, size=state.shape)

        # we keep all the histories on the device
        # note that the latest action and reward will be "padding"
        states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
        actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
        rewards = torch.zeros(0, device=device, dtype=torch.float32)

        target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
        timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

        sim_states = []

        episode_return, episode_length = 0, 0

        env.reset()
        for t in range(max_ep_len):

            # add padding
            actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
            rewards = torch.cat([rewards, torch.zeros(1, device=device)])

            action = model.get_action(
                (states.to(dtype=torch.float32) - state_mean) / state_std,
                actions.to(dtype=torch.float32),
                rewards.to(dtype=torch.float32),
                target_return.to(dtype=torch.float32),
                timesteps.to(dtype=torch.long),
            )
            actions[-1] = action
            action = action.detach().cpu().numpy()

            state, reward, done, info = env.step(action)

            cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
            states = torch.cat([states, cur_state], dim=0)
            rewards[-1] = reward

            if mode != 'delayed':
                pred_return = target_return[0,-1] - (reward/scale)
            else:
                pred_return = target_return[0,-1]
            target_return = torch.cat(
                [target_return, pred_return.reshape(1, 1)], dim=1)
            timesteps = torch.cat(
                [timesteps,
                torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

            episode_return += reward
            episode_length += 1

            if done or info['success']:
                success += 1
                print(f"current success: {success}")
                break
        episode_return_list.append(episode_return)
        episode_length_list.append(episode_length)
    

    return np.mean(np.array(episode_return)), np.mean(np.array(episode_length)), float(success/20)
