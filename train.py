'''
Author: Jikun Kang
Date: 1969-12-31 19:00:00
LastEditTime: 2023-05-14 16:57:49
LastEditors: Jikun Kang
FilePath: /Hyper-DT/train.py
'''

import random
from src.load_dataset import get_batch
from src.seq_trainer import SequenceTrainer
from src.create_self_dataset import create_self_dataset, prepare_dataset
from src.minlora import add_lora, get_lora_params
import namegenerator
import argparse
import tqdm
import os
import socket
from pathlib import Path
import time
from typing import Optional
import torch
import numpy as np
import wandb
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from src.env_wrapper import build_env_fn
from src.create_dataset import create_dataset
from src.env_utils import ATARI_NUM_ACTIONS, ATARI_NUM_REWARDS, ATARI_RETURN_RANGE
from src.decision_transformer import DecisionTransformer
from torch.utils.data import Dataset
from src.trainer import Trainer
import metaworld

meta_world_game_list = ['assembly-v2', 'basketball-v2', 'button-press-topdown-v2', 
'button-press-topdown-wall-v2', 'button-press-v2', 'button-press-wall-v2', 'coffee-button-v2', 'coffee-pull-v2', 'coffee-push-v2', 'dial-turn-v2', 'disassemble-v2', 'door-close-v2', 'door-open-v2', 'drawer-close-v2', 'drawer-open-v2', 'faucet-open-v2', 'faucet-close-v2', 'hammer-v2', 'handle-press-side-v2', 'handle-press-v2', 'handle-pull-side-v2', 'handle-pull-v2', 'lever-pull-v2', 'peg-insert-side-v2', 'pick-place-wall-v2', 'pick-out-of-hole-v2', 'reach-v2', 'push-back-v2', 'push-v2', 'pick-place-v2', 'plate-slide-v2', 'plate-slide-side-v2', 'plate-slide-back-v2', 'plate-slide-back-side-v2', 'peg-unplug-side-v2', 'soccer-v2', 'stick-push-v2', 'stick-pull-v2', 'push-wall-v2', 'reach-wall-v2', 'shelf-place-v2', 'sweep-into-v2', 'sweep-v2', 'window-open-v2', 'window-close-v2']

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'n', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


class StateActionReturnDataset(Dataset):

    def __init__(
        self,
        obs,
        block_size,
        actions,
        done_idxs,
        rtgs,
        timesteps,
        rewards
    ):
        self.block_size = block_size
        # self.vocab_size = max(actions) + 1
        self.obs = obs
        self.actions = actions
        self.done_idxs = done_idxs
        self.rtgs = rtgs
        self.timesteps = timesteps
        self.rewards = rewards

    def __len__(self):
        return len(self.obs) - self.block_size

    def __getitem__(self, idx):
        block_size = self.block_size // 3
        done_idx = idx + block_size
        for i in self.done_idxs:
            if i > idx:  # first done_idx greater than idx
                done_idx = min(int(i), done_idx)
                break
        idx = done_idx - block_size
        if idx < 0:
            idx = 0
            done_idx = idx + block_size

        states = self.obs[idx:done_idx].to(
            dtype=torch.float32)  # .reshape(block_size, -1)  # (block_size, 3*64*64)
        states = states / 255.
        actions = self.actions[idx:done_idx].to(
            dtype=torch.long).unsqueeze(1)  # (block_size, 1)
        rtgs = self.rtgs[idx:done_idx].to(
            dtype=torch.float32).unsqueeze(1)
        # timesteps = torch.tensor(
        #     self.timesteps[idx:idx+1], dtype=torch.int64).unsqueeze(1)
        rewards = self.rewards[idx:done_idx].to(
            dtype=torch.float32).unsqueeze(1)

        return states, rtgs, actions, rewards


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run(args):
    # set saving directory
    if args.folder_prefix is not None:
        folder_path = args.folder_prefix+'/MDT_results'
    else:
        folder_path = 'MDT_results'
    # run_dir = Path(folder_path) / args.game_name / \
    #     args.experiment_name / namegenerator.gen()
    run_dir = Path(folder_path) / namegenerator.gen()
    print(f"The run dir is {str(run_dir)}")
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # set seed
    set_seed(args.seed)

    # Init Logger
    if args.use_wandb:
        run = wandb.init(
            config=args,
            project=args.experiment_name,
            entity=args.user_name,
            notes=socket.gethostname(),
            name=f"seed_{str(args.seed)}",
            group=args.game_name,
            dir=str(run_dir),
            job_type='training',
            reinit=True,
            mode=args.wandb_status,
        )
        logger = None
    else:
        logger = SummaryWriter(run_dir)

    # init model
    dt_model = DecisionTransformer(
        state_dim=39,
        act_dim=4,
        max_length=20,
        max_ep_len=1000,
        hidden_size=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_inner=4*args.n_embd,
        activation_function='relu',
        n_positions=1024,
        resid_pdrop=args.resid_drop,
        attn_pdrop=args.attn_drop,
    )

    dt_model.to(device=args.device)
    if args.device == 'cuda':
        if args.n_gpus:
            dt_model = nn.DataParallel(dt_model)

    # init eval game list
    ml45 = metaworld.ML45() # Construct the benchmark, sampling tasks

    eval_game_list = []
    eval_game_name = []
    for name, env_cls in ml45.test_classes.items():
        env= env_cls() 
        task = random.choice([task for task in ml45.test_tasks
                                if task.env_name == name])
        env.set_task(task)
        eval_game_list.append(env)
        eval_game_name.append(name)

    # init train_dataset
    train_dataset_list = []
    train_game_list = []
    # TODO: fix this
    meta_world_game_list = []
    if not args.eval:
        optimizer = torch.optim.AdamW(
            dt_model.parameters(),
            lr=args.optimizer_lr,
            weight_decay=args.weight_decay,
        )
        trainer_list = []
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda steps: min((steps+1)/args.warmup_steps, 1)
        )
        if args.apply_lora:
            meta_world_game_list = ['bin-picking-v2']
            for name, env_cls in ml45.test_classes.items():
                if name in meta_world_game_list:
                    env= env_cls() 
                    task = random.choice([task for task in ml45.test_tasks
                                            if task.env_name == name])
                    env.set_task(task)
                    print(f"======>Loading Game {name}")
                    num_traj, p_sample, traj, sorted_inds, state_mean, state_std = prepare_dataset(name)

                    trainer = SequenceTrainer(
                        model=dt_model,
                        optimizer=optimizer,
                        batch_size=args.batch_size,
                        get_batch=get_batch,
                        scheduler=scheduler,
                        loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
                        num_trajectories=num_traj,
                        p_sample=p_sample,
                        trajectories=traj,
                        sorted_inds=sorted_inds,
                        state_mean=state_mean,
                        state_std=state_std,
                        run_dir=run_dir,
                        eval_envs=eval_game_list,
                        eval_name_list=eval_game_name,
                        train_envs=[env],
                        train_name_list=[name],
                        device=args.device
                    )
                    trainer_list.append(trainer)
        else:
            meta_world_game_list = ['basketball-v2', 
                                    'button-press-topdown-v2', 
            'button-press-topdown-wall-v2', 
            'button-press-v2', 'button-press-wall-v2', 'coffee-button-v2', 
            'coffee-pull-v2', 'dial-turn-v2', 'disassemble-v2', 'door-open-v2', 
            'drawer-close-v2', 'drawer-open-v2', 'faucet-open-v2', 'faucet-close-v2', 
            'hammer-v2', 'handle-press-side-v2', 'handle-press-v2', 'handle-pull-side-v2', 
            'handle-pull-v2', 'lever-pull-v2', 'plate-slide-v2', 
            'plate-slide-side-v2', 'plate-slide-back-v2', 'plate-slide-back-side-v2', 
            'peg-unplug-side-v2', 'soccer-v2', 'stick-push-v2', 'stick-pull-v2', 'window-open-v2', 'window-close-v2']
            for name, env_cls in ml45.train_classes.items():
                if name in meta_world_game_list:
                    env= env_cls() 
                    task = random.choice([task for task in ml45.train_tasks
                                            if task.env_name == name])
                    env.set_task(task)
                    print(f"======>Loading Game {name}")
                    num_traj, p_sample, traj, sorted_inds, state_mean, state_std = prepare_dataset(name)

                    trainer = SequenceTrainer(
                        model=dt_model,
                        optimizer=optimizer,
                        batch_size=args.batch_size,
                        get_batch=get_batch,
                        scheduler=scheduler,
                        loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
                        num_trajectories=num_traj,
                        p_sample=p_sample,
                        trajectories=traj,
                        sorted_inds=sorted_inds,
                        state_mean=state_mean,
                        state_std=state_std,
                        run_dir=run_dir,
                        eval_envs=eval_game_list,
                        eval_name_list=eval_game_name,
                        train_envs=[env],
                        train_name_list=[name],
                        device=args.device
                    )
                    trainer_list.append(trainer)
    
    # totoal params
    total_params = sum(params.numel() for params in dt_model.parameters())
    print(f"======> Total number of params are {total_params}")
    if args.load_path != '0':
        # best_model = wandb.restore('tf_model.pt','jaxonkang/meta-world/z0wfhfhl')
        epoch, loss = trainer.load_model(args.load_path, args.apply_lora)
        print(f"========> Load CKPT from {args.load_path}")
        epoch = epoch+1
    else:
        epoch = 0
    
    if args.apply_lora:
        pass
        # print("========>Adding LoRA")
        # add_lora(dt_model)
        # parameters = [
        #     {"params": list(get_lora_params(dt_model))}
        # ]
        # optimizer = torch.optim.AdamW(parameters, lr=args.optimizer_lr)
        # total_params = sum(params.numel() for params in dt_model.parameters())
        # print(f"======> Total number of params after LoRA {total_params}")
    else:
        optimizer = torch.optim.AdamW(
            dt_model.parameters(),
            lr=args.optimizer_lr,
            weight_decay=args.weight_decay,
        )

    if args.train:
        print("========>Start Training")
        for t in range(args.max_epochs):
            for trainer in trainer_list:
                if args.apply_lora:
                    pass
                    # trainer.set_optimizer(optimizer=optimizer)
                outputs=trainer.train_iteration(
                    num_steps=10000,
                    iter_num=t+1,
                    print_logs=True
                )
                # logs= trainer.evaluate()
                if args.use_wandb:
                    wandb.log(outputs)
                        # wandb.log(logs)
    elif args.eval:
        print("========>Start Evaluation")
        logs= trainer.evaluate()
        if args.use_wandb:
            wandb.log(logs)
    else:
        NotImplementedError("No actions for training or evaluation")

    # close logger
    if args.use_wandb:
        run.finish()
    else:
        logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model configs
    parser.add_argument('--n_embd', type=int, default=512)  # 1280
    parser.add_argument('--n_layer', type=int, default=4)  # 10
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--seq_len', type=int, default=28)
    parser.add_argument('--attn_drop', type=float, default=0.1)
    parser.add_argument('--resid_drop', type=float, default=0.1)
    parser.add_argument('--create_hnet', type=str2bool, default=False)
    parser.add_argument('--use_gw', type=str2bool, default=False)

    # Logging configs
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--use_wandb', type=str2bool, default=False)
    parser.add_argument("--user_name", type=str, default='jaxonkang',
                        help="[for wandb usage], to specify user's name for simply collecting training data.")
    parser.add_argument("--n_gpus", action='store_true', default=False)

    # Training configs
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--steps_per_iter', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--training_samples', type=int, default=1000)
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--apply_lora', type=str2bool, default=False)
    parser.add_argument("--train", type=str2bool, default=False)

    # Evaluation configs
    parser.add_argument('--eval_steps', type=int, default=1000)
    parser.add_argument('--eval_game_list', nargs='+', default=[])
    parser.add_argument('--num_eval_envs', type=int, default=1)
    parser.add_argument('--eval_freq', type=int, default=10)
    parser.add_argument("--eval", type=str2bool, default=False)

    # Optimizer configs
    parser.add_argument('--optimizer_lr', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--grad_norm_clip', type=float, default=1.)
    parser.add_argument('--num_workers', type=int, default=0)

    # Dataset related
    parser.add_argument('--data_steps', type=int, default=500000)
    parser.add_argument('--num_buffers', type=int, default=50)
    parser.add_argument('--game_name', type=str, default='Amidar')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--trajectories_per_buffer', type=int, default=10,
                        help='Number of trajectories to sample from each of the buffers.')
    parser.add_argument('--train_game_list', nargs='+', default=[])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_datasets', type=int, default=1)
    parser.add_argument('--folder_prefix', type=str, default=None)

    parser.add_argument("--save_freq", default=10, type=int)
    parser.add_argument('--experiment_name', default='metaworld', type=str)
    parser.add_argument('--cuda_cores', type=str, default=None)
    parser.add_argument('--wandb_status', type=str, default='online')

    args = parser.parse_args()
    if args.cuda_cores is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_cores
    run(args)
