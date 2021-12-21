import multiprocessing as mp
import argparse

import numpy as np
import torch
from gym_minigrid.env_reader import read_map
from spinup.algos.pytorch.sac.ambiguity import OnlineACAmbiguityAgent

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


def run_online_ac_ambiguity(num_env, policy_type='softmax', discrete=True, adaptive_pruning_constant=-100,
                            pruning_decay=0.95, tau_constant=1, tau_decay=0.975, hyper_param_study=None):
    the_train_env, map_name = read_map(num_env, random_start=False, discrete=discrete)
    the_test_env, map_name = read_map(num_env, random_start=False, discrete=discrete)

    if hyper_param_study == 'pruning_constant':
        experiment_name = f'{map_name}{num_env}-online-ac-{policy_type}-pruning-constant={adaptive_pruning_constant}'
    elif hyper_param_study == 'tau_constant':
        experiment_name = f'{map_name}{num_env}-online-ac-{policy_type}-tau-constant={tau_constant}'
    elif hyper_param_study == 'tau_decay':
        experiment_name = f'{map_name}{num_env}-online-ac-{policy_type}-tau-decay={tau_decay}'
    elif hyper_param_study == 'pruning_decay':
        experiment_name = f'{map_name}{num_env}-online-ac-{policy_type}-pruning-decay={pruning_decay}'
    else:
        experiment_name = f'{map_name}{num_env}-online-ac-{policy_type}'

    if num_env in {1, 2, 3, 4, 9, 10, 11, 12, 17, 18, 19, 20, 25, 26, 27, 28, 33, 34, 35, 36}:
        all_models = [None, None, None]
        all_model_names = ['rg', 'fg1', 'fg2']
    else:
        all_models = [None, None, None, None, None]
        all_model_names = ['rg', 'fg1', 'fg2', 'fg3', 'fg4']

    max_ep_length = 49 ** 2

    agent = OnlineACAmbiguityAgent(
        state_space=the_train_env.observation_space,
        action_space=the_train_env.action_space,
        name=f'OnlineACContinuous',
        all_models=all_models,
        all_model_names=all_model_names,
        env=the_test_env,
        real_goal_pruning_constant=0,
        policy=policy_type,
        adaptive_pruning_constant=adaptive_pruning_constant,
        pruning_decay=pruning_decay,
        experiment_name=experiment_name,
        discrete=discrete,
        max_ep_len=max_ep_length,
        batch_size=128,
        hidden_dim=64,
        discount_rate=0.975,
        alpha=0.2,
        lr_decay=0.95,
        q_gain_pruning_constant=0,
        start_steps=40000,
        tau=tau_constant,
        tau_decay=tau_decay,
        steps_per_epoch=10000,
        critic_lr=1e-3,
        pi_lr=1e-3,
        num_epochs=120,
        num_test_eps=1,
    )

    agent.train(the_train_env)


def train_online_ambiguity_vs_pruning_decay(map_num, discrete=True):
    policy = 'softmax'
    decay_params = [1, 0.975, 0.95, 0.9, 0.75, 0.5]
    adaptive_pruning_constant = -100
    tau_constant = 1.0
    tau_decay = 0.975
    pool = mp.Pool(len(decay_params))
    pool.starmap(run_online_ac_ambiguity,
                 [(map_num, policy, discrete, adaptive_pruning_constant, decay_param, tau_constant, tau_decay,
                   'pruning_decay') for decay_param in decay_params])


def train_online_ambiguity_vs_pruning_constant(map_num, discrete=True):
    policy = 'softmax'
    pruning_constants = [-1, -10, -50, -100, -500, -1000]
    decay_param = 0.95
    tau_constant = 1.0
    tau_decay = 0.975
    pool = mp.Pool(len(pruning_constants))
    pool.starmap(run_online_ac_ambiguity,
                 [(map_num, policy, discrete, pruning_constant, decay_param, tau_constant, tau_decay,
                   'pruning_constant') for pruning_constant in pruning_constants])


def train_online_ambiguity_vs_tau_constant(map_num, discrete=True):
    policy = 'softmax'
    tau_constants = [0, 0.5, 1.0, 2.0, 5.0, 10.0]
    tau_decay = 0.975
    decay_param = 0.90
    adaptive_pruning_constant = -10
    pool = mp.Pool(len(tau_constants))
    pool.starmap(run_online_ac_ambiguity,
                 [(map_num, policy, discrete, adaptive_pruning_constant, decay_param, tau_constant, tau_decay,
                   'tau_constant') for tau_constant in tau_constants])


def train_online_ambiguity_vs_tau_decay(map_num, discrete=True):
    policy = 'softmax'
    tau_constant = 1.0
    tau_decays = [1, 0.99, 0.975, 0.95, 0.9, 0.5]
    decay_param = 0.90
    adaptive_pruning_constant = -10
    pool = mp.Pool(len(tau_decays))
    pool.starmap(run_online_ac_ambiguity,
                 [(map_num, policy, discrete, adaptive_pruning_constant, decay_param, tau_constant, tau_decay,
                   'tau_decay') for tau_decay in tau_decays])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hyperparam', type=str, default='pruning_constant')
    parser.add_argument('--map_num', type=int, default=1)
    parser.add_argument('--policy', type=str, default='softmax')
    parser.add_argument('--discrete', type=bool, default=False)
    args = parser.parse_args()

    map_num = args.map_num
    discrete = args.discrete
    policy = args.policy
    hyperparam = args.hyperparam
    if policy == 'softmax':
        if hyperparam == 'pruning_constant':
            train_online_ambiguity_vs_pruning_constant(map_num=map_num, discrete=discrete)
        elif hyperparam == 'pruning_decay':
            train_online_ambiguity_vs_pruning_decay(map_num=map_num, discrete=discrete)
        elif hyperparam == 'tau_constant':
            train_online_ambiguity_vs_tau_constant(map_num=map_num, discrete=discrete)
        elif hyperparam == 'tau_decay':
            train_online_ambiguity_vs_tau_decay(map_num=map_num, discrete=discrete)
        else:
            raise ValueError("Invalid hyperparam type")
    elif policy == 'hardmax':
        run_online_ac_ambiguity(num_env=map_num,
                                policy_type=policy,
                                discrete=discrete)
    else:
        raise ValueError("Invalid policy type")