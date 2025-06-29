import os
import heapq

import numpy as np
from tensorboard.backend.event_processing import event_accumulator

from src.envs.trainer import (
    Trainer,
    BRONZE_MAZES,
    WOOD_MAZES,
)
from src.game.template import (
    DEFAULT_KUTULU_ACTIONS,
    EXTENDED_KUTULU_ACTIONS,
)


def get_trainer(
    num_experiments,
    league_level,
    gamma, size, lr,
    fc_dim, conv_dim, 
    random_epsilon,
    sanity_coef, reward_for_win, reward_for_lose,
    entropy_coef, value_loss_coef,
    clip_ratio,
    ppo_epochs, mini_batch_size,
    target_kl, max_grad_norm,
    gae_lambda, num_envs,
    good_plan_bonus, bad_plan_bonus, good_light_bonus, bad_light_bonus,
    verbose=False, silent=False,
):
    mazes = BRONZE_MAZES if league_level >= 3 else WOOD_MAZES
    actions = EXTENDED_KUTULU_ACTIONS if league_level >= 3 else DEFAULT_KUTULU_ACTIONS
    action_space_n = len(actions)
    
    env_kwargs = {
        'reward_params': {
            'sanity_coef': sanity_coef, 'reward_for_win': reward_for_win, 'reward_for_lose': reward_for_lose
        }
    }
    random_agent_info = {
        'train': False,
        'type': 'epsilon_wait',
        'action_space_n': action_space_n,
        'epsilon_params': {'start': random_epsilon, 'final': random_epsilon, 'decay': int(4 * 10**5)},
        'state_type': 'closest',
        'action': 'WAIT',
    }
    research_agent_info = {
        'train': True,
        'type': 'ppo',
        'action_space_n': action_space_n,
        'actions': actions,
        'state_type': 'conv',
        'model_params': {
            'fc_dim': fc_dim,
            'conv_dim': conv_dim,
            'size': size,
        },
        'gamma': gamma,
        'lr': lr,
        'optimizer': 'adamw',
        'scheduler_params': {'type': 'cosine', 'T_max': num_experiments},
        'entropy_coef': entropy_coef,
        'value_loss_coef': value_loss_coef,
        'clip_ratio': clip_ratio,
        'ppo_epochs': ppo_epochs,
        'mini_batch_size': mini_batch_size,
        'target_kl': target_kl,
        'max_grad_norm': max_grad_norm,
        'gae_lambda': gae_lambda,
        'reward_params': {
            'good_plan_bonus': good_plan_bonus,
            'bad_plan_bonus': bad_plan_bonus,
            'good_light_bonus': good_light_bonus,
            'bad_light_bonus': bad_light_bonus,
        }
    }
    agents_info = [research_agent_info]
    for i in range(len(agents_info), 4):
        agents_info.append(dict(random_agent_info))
    assert len(agents_info) == 4
    trainer = Trainer(
        num_experiments=num_experiments, agents_info=agents_info, shuffle=True,
        league_level=league_level, mazes=mazes, actions=actions, log_dir='../runs', verbose=verbose,
        env_kwargs=env_kwargs, silent=silent, num_envs=num_envs,
    )
    return trainer


def get_score(result, is_ppo=True):
    agent_id = 0
    metrics_list = result[2]
    acc = [
        [
            metrics[key][agent_id][0]
            for key in [
                'check_exp_normal', 'check_exp_coridor', 'check_exp_corner',
                'check_wan_normal', 'check_wan_coridor', 'check_wan_corner',
            ]
        ] for metrics in metrics_list[-10:]
    ]
    top_action_cnt = [
        [
            metrics[key][agent_id][2]
            for key in [
                'check_exp_normal',
                'check_wan_normal',
            ]
        ] for metrics in metrics_list[-10:]
    ]
    mean_q = [
        [
            metrics[key][agent_id][4]
            for key in [
                'check_exp_normal',
                'check_wan_normal',
            ]
        ] for metrics in metrics_list[-10:]
    ]
    if is_ppo:
        value_loss = [
            metrics['value_loss'][agent_id]
            for metrics in metrics_list[-10:]
        ]
        score = np.mean(acc) + 0.25 * np.mean(top_action_cnt) - 0.5 * np.mean(value_loss)
    else:
        score = np.mean(acc) + 0.25 * np.mean(top_action_cnt) - 0.1 * np.max(np.max(mean_q, axis=0) - np.min(mean_q, axis=0))
    return score


def get_best_iter(datetime_start, model_type):
    l = str(datetime_start)
    left = l.split()[0].replace('-', '')
    right_orig = l.split()[1].replace(':', '')[:6]
    ea = None
    for delta in range(0, 100):
        try:
            right = f'{int(right_orig) + delta:06d}'
            exp_name = f'{left}-{right}'
            logdir = f'../runs/{exp_name}/agent0_{model_type}'
            logfname = os.listdir(logdir)[0]
            ea = event_accumulator.EventAccumulator(f'{logdir}/{logfname}')
            break
        except FileNotFoundError:
            continue
    assert ea
    ea.Reload()

    scalars_top_a = [
        ea.Scalars(k) for k in [
            'Check/Explorer/top_a',
            'Check/Wanderer/top_a',
        ]
    ]
    scalars_acc = [
        ea.Scalars(k) for k in [
            'Check/Explorer/acc',
            'Check/Wanderer/acc',
            'Check/Explorer/acc_coridor',
            'Check/Wanderer/acc_coridor',
            'Check/Explorer/acc_corner',
            'Check/Wanderer/acc_corner',
        ]
    ]

    st = {
        st_top_a[0].step: np.mean([v.value for v in st_acc]) + 0.25 * np.mean([v.value for v in st_top_a])
        for st_top_a, st_acc in list(zip(list(zip(*scalars_top_a)), list(zip(*scalars_acc))))
        if st_top_a[0].step % 100 == 0
    }

    max_k = list(st.keys())[0]
    max_v = st[max_k]
    for k, v in st.items():
        if v >= max_v:
            max_v = v
            max_k = k

    return exp_name, max_k, max_v

def get_top_k(agents_info, num_experiments, league_level, mazes, actions, k=2):
    trainer = Trainer(
        num_experiments=num_experiments, agents_info=agents_info, shuffle=True,
        league_level=league_level, mazes=mazes, actions=actions, log_dir='../runs', verbose=False,
        silent=True,
    )
    result = trainer.train()
    reward_list = result[0]

    winner_stats = {}
    for rewards in reward_list:
        scores = np.sum(~np.isnan(rewards), axis=0)
        for i, score in enumerate(scores):
            if score == np.max(scores):
                winner_stats[i] = winner_stats.get(i, 0) + 1
    top_k = heapq.nlargest(2, winner_stats, key=winner_stats.get)
    best_agents_info = [agents_info[k] for k in top_k]
    return best_agents_info, top_k, winner_stats
