import os
import heapq

import numpy as np
from tensorboard.backend.event_processing import event_accumulator

from src.envs.trainer import Trainer


def get_score(result):
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
    return best_agents_info, winner_stats
