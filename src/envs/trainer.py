import os
import json
from copy import deepcopy
from datetime import datetime
from tqdm import tqdm

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from src.envs.kutulu_world import (
    KutuluWorldEnv,
    EXTENDED_KUTULU_ACTIONS,
    DEFAULT_KUTULU_ACTIONS,
)
from src.envs.agent_validator import AgentValidator
from src.envs.agents.dqn_agent import DQNAgentBase
from src.envs.agents.ppo_agent import PPOAgent
from src.envs.agents.nn_agent import NNAgent
from src.envs.agents.actor_agent import ActorAgent
from src.envs.agents.agent_factory import get_agent

WOOD_MAZES = [
    "PacMan",
    "Hypersonic",
    "Oasis",
    "Corridors",
    "OriginOfSymmetry",
    "FourOfAKind",
]

BRONZE_MAZES = [
    "FourOfAKind",
    "Hypersonic",
    "OriginOfSymmetry",
    "ShelterMe",
    "SlasherHell",
    "Typhoon",
    "Oasis",
    "Cross",
    "ShelterInPeril",
    "Corridors",
    "Roommates",
    "PacMan",
    "ChallengeFromBeyond",
    "Cog",
    "HillClimbing",
    "Pixelated",
]


def get_mazes_by_league(league_level):
    return BRONZE_MAZES if league_level >= 4 else WOOD_MAZES

def get_actions_by_league(league_level):
    return EXTENDED_KUTULU_ACTIONS if league_level >= 3 else DEFAULT_KUTULU_ACTIONS


class Trainer:
    def __init__(self, num_experiments, agents_info, league_level,
                 agents=None,
                 env_kwargs=None, shuffle=True,
                 log_dir='../../../runs', output_dir='../../../output',
                 exp_name=None,
                 verbose=False, silent=False, only_train=True, asc_difficulty=False,
                 num_envs=1, use_tqdm=True, metrics_int=10, seed=None):
        self.num_experiments = num_experiments
        self.league_level = league_level
        self.mazes = get_mazes_by_league(league_level)
        self.actions = get_actions_by_league(league_level)
        self.env_kwargs = env_kwargs or {}
        self.shuffle = shuffle
        self.verbose = verbose
        self.silent = silent
        self.asc_difficulty = asc_difficulty
        self.use_tqdm = use_tqdm
        self.metrics_int = metrics_int

        self.num_envs = num_envs

        if not self.silent:
            if exp_name is None:
                exp_name = datetime.now().strftime('%Y%m%d-%H%M%S')
            self.log_dir = os.path.join(log_dir, exp_name)
            self.output_dir = output_dir
            
            date, time = str(datetime.now()).split()
            date_dir = f'{self.output_dir}/{date}'
            os.makedirs(date_dir, exist_ok=True)
            
            self.checkpoints_dir = os.path.join(date_dir, exp_name)
            os.makedirs(self.checkpoints_dir, exist_ok=True)

            if agents_info is not None:
                with open(f'{self.checkpoints_dir}/agents_info.json', 'w') as f:
                    json.dump(agents_info, f)

        self.agent_validator = AgentValidator(self.actions)
        self.agent_validator_plan = AgentValidator(self.actions, player_params=(100, 1, 0))
        self.only_train = only_train
        self.seed = seed

        if agents is not None:
            self.agents = agents
        else:
            self.agents = []
            for i, agent_info in enumerate(agents_info):
                agent_info = dict(agent_info)
                _name = None
                if 'name' in agent_info:
                    _name = agent_info.pop('name')
                _type = agent_info['type']
                agent_info = deepcopy(agent_info)
                agent_info['verbose'] = self.verbose
                agent = get_agent(agent_info)
                self.agents.append(agent)

        if not self.silent:
            for i, agent in enumerate(self.agents):
                if not agent.train:
                    continue
                agent_dir = f'agent{i}'
                agent_log_dir = os.path.join(self.log_dir, agent_dir)
                agent.writer = SummaryWriter(log_dir=agent_log_dir)
                if self.verbose:
                    print(f"agent {agent_dir}, type: {type(agent)}")

        self.players_count = len(self.agents)
        self.agent_map = np.arange(len(self.agents))
        
        # Initialize multi-environment support for PPO agents
        if self.num_envs > 1:
            for agent in self.agents:
                if isinstance(agent, PPOAgent):
                    agent.init_multi_env(self.num_envs)
    
    def reset_env(self, seed=None, maze_name=None, step=None, port=8080):
        if maze_name is None:
            if seed is not None:
                maze_name = np.random.RandomState(seed).choice(self.mazes).item()
            else:
                maze_name = np.random.choice(self.mazes).item()
        if self.asc_difficulty and step is not None:
            league_level = self.league_level * (step - 1) // self.num_experiments + 1
            if self.verbose:
                print(f'current league_level: {league_level}')
        else:
            league_level = self.league_level
        env = KutuluWorldEnv(
            server_host=f'localhost:{port}',
            maze_name=maze_name,
            league_level=league_level,
            players_count=self.players_count,
            actions=self.actions,
            **self.env_kwargs
        )
        observation, info = env.reset(seed=seed)

        for agent in self.agents:
            agent.set_env(env)

        return env
    
    def play_rollout(self, seed=None, step=None):
        if self.num_envs == 1:
            return self.play_single_rollout(seed, step=step)
        else:
            return self.play_multi_rollout(seed, step=step)

    def play_single_rollout(self, seed=None, maze_name=None, env_idx=None, step=None, only_eval=False):
        env = self.reset_env(seed, maze_name, step)
        if self.verbose:
            if env_idx is not None:
                print(f"Environment {env_idx}: {maze_name} (seed: {seed})")
            else:
                print(f"Environment: {maze_name} (seed: {seed})")
        for agent in self.agents:
            agent.set_env(env)
        rollout_rewards = []
        game_over = False
        if self.shuffle:
            np.random.shuffle(self.agent_map)
        step = 0
        while not game_over:
            assert env.turn == step
            step += 1
            # action = env.sample_valid_action()
            # WAIT
            action = [4 for _ in range(len(self.agents))]

            for player_id in env.active_players():
                agent_id = self.agent_map[player_id]
                if only_eval:
                    At = self.agents[agent_id].inference_step(player_id)
                else:
                    _, At = self.agents[agent_id].generate_state_and_step(player_id)
                action[player_id] = At

            entities, rewards, game_over, info = env.step(action)
            rollout_rewards.append(rewards)

            if self.verbose:
                rewards_by_agent = [rewards[i] for i in np.argsort(self.agent_map)]
                actions_by_agent = [self.actions[action[i]] for i in np.argsort(self.agent_map)]
                if env_idx is not None:
                    print(f"Env {env_idx}, step: {step}, rewards: {rewards_by_agent}")
                    print(f"Env {env_idx}, step: {step}, actions: {actions_by_agent}")
                else:
                    print(f"step: {step}, rewards: {rewards_by_agent}")
                    print(f"step: {step}, actions: {actions_by_agent}")
                # env.viz_map()
            
            if only_eval:
                continue

            train_agents_game_over = True
            for player_id, reward in enumerate(rewards):
                agent_id = self.agent_map[player_id]
                agent = self.agents[agent_id]
                if agent.train:
                    agent_game_over = env.is_game_over_for_player(player_id) or game_over
                    other_rewards = list(rewards)
                    other_rewards.pop(player_id)
                    agent.append_observation(player_id, reward, agent_game_over, env_idx, other_rewards)
                    need_train_step = False
                    train_agents_game_over = train_agents_game_over and agent_game_over
                    if isinstance(agent, DQNAgentBase):
                        need_train_step = agent_game_over or (player_id in env.active_players())
                    elif hasattr(agent, 'train_multi_env_step'):
                        need_train_step = agent_game_over and env_idx is None
                    elif isinstance(agent, ActorAgent):
                        need_train_step = agent_game_over
                    if need_train_step:
                        if self.verbose:
                            print(f"step: {step}, agent_id: {agent_id}, type: {str(type(agent)).split('.')[-1][:-2]}, "
                                f"train_step, reward: {reward}, game_over: {agent_game_over}")
                        agent.train_step()
            if self.only_train:
                game_over = train_agents_game_over
        
        if self.verbose:
            if env_idx is not None:
                print(f"Environment {env_idx} finished after {step} steps")
            else:
                print(f"Environment finished after {step} steps")
        rollout_rewards = np.array(rollout_rewards, dtype=float)[:,np.argsort(self.agent_map)]
        return rollout_rewards

    def play_multi_rollout(self, seed=None, step=None):
        """Play rollout across multiple environments simultaneously"""
        # Track rewards for each environment
        env_rollout_rewards = [[] for _ in range(self.num_envs)]

        # Select different mazes for diversity
        selected_mazes = self.mazes[:self.num_envs] if self.num_envs <= len(self.mazes) else \
                        (self.mazes * ((self.num_envs // len(self.mazes)) + 1))[:self.num_envs]

        for env_idx in range(self.num_envs):
            maze_name = selected_mazes[env_idx]
            env_seed = seed + env_idx if seed is not None else None
            rollout_rewards = self.play_single_rollout(env_seed, maze_name, env_idx, step)
            env_rollout_rewards[env_idx] = rollout_rewards

        # Trigger training for multi-environment agents after all environments complete
        for agent in self.agents:
            if agent.train and hasattr(agent, 'train_multi_env_step'):
                if self.verbose:
                    print(f"Training agent {self.agents.index(agent)} with multi-env data")
                agent.train_multi_env_step()

        # Combine rewards from all environments
        # For compatibility, return the average rewards across environments
        max_steps = max(len(rewards) for rewards in env_rollout_rewards)
        combined_rewards = np.array([
            np.vstack((
                rewards,
                np.repeat(np.zeros((1, rewards.shape[1])), max_steps - rewards.shape[0], axis=0)
            ))
            for rewards in env_rollout_rewards
        ])
        combined_rewards = np.nanmean(combined_rewards, axis=0)
        return combined_rewards

    def save_models(self, step):
        for i,agent in enumerate(self.agents):
            checkpoint_dir = f'{self.checkpoints_dir}/agent{i}'
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_dir_step = f'{checkpoint_dir}/{step}'
            os.makedirs(checkpoint_dir_step, exist_ok=True)
            agent.save_agent(checkpoint_dir_step)
        return self.checkpoints_dir

    def train(self):
        reward_list = []
        model_dir_list = []
        metrics_list = []

        exps = range(self.num_experiments)
        if not self.verbose and self.use_tqdm:
            exps = tqdm(exps)
        for i in exps:
            step = i + 1
            if self.verbose:
                print()
                print(f"rollout: {step}")
            if self.seed:
                seed = np.random.RandomState(self.seed + step).randint(999999)
            else:
                seed = None
            rollout_rewards = self.play_rollout(step=step, seed=seed)
            reward_list.append(rollout_rewards)

            # # Save models periodically
            # if not self.silent and step % save_models_int == 0:
            #     model_dir = self.save_models(step)
            #     model_dir_list.append(model_dir)

            # Log metrics periodically
            if step % self.metrics_int == 0:
                metrics = self._calculate_metrics(reward_list[-100:])
                metrics_list.append(metrics)
                if not self.silent:
                    self._log_metrics(step, metrics)

        if not self.silent:
            for i, agent in enumerate(self.agents):
                if not agent.train:
                    continue
                agent.writer.close()
        return reward_list, model_dir_list, metrics_list
    
    def _calculate_metrics(self, reward_list):
        metrics = {}
        av = self.agent_validator
        av_plan = self.agent_validator_plan
        total_reward_list = np.array([np.nansum(rewards, axis=0) for rewards in reward_list])
        metrics['rewards'] = np.mean(total_reward_list, axis=0)
        winner_list = np.argmax(total_reward_list, 1)
        metrics['winner_list'] = [np.sum(winner_list == i) / len(winner_list) for i, agent in enumerate(self.agents)]
        metrics['check_policy'] = [agent.check_policy() for agent in self.agents]
        metrics['eps'] = [agent.get_eps() for agent in self.agents]
        metrics['check_exp'] = [
            av.check_entity_nearby(agent, 'EXPLORER', n_min=2, n_max=3, env_types=('normal', 'coridor', 'corner'))
            for agent in self.agents
        ]
        metrics['check_wan'] = [
            av.check_entity_nearby(agent, 'WANDERER', n_min=1, n_max=2, env_types=('normal', 'coridor', 'corner'))
            for agent in self.agents
        ]
        # metrics['check_exp_normal'] = [av.check_entity_nearby(agent, 'EXPLORER', n_min=2, n_max=3) for agent in self.agents]
        # metrics['check_wan_normal'] = [av.check_entity_nearby(agent, 'WANDERER', n_min=1, n_max=2) for agent in self.agents]
        metrics['check_exp_normal_plan1'] = [av_plan.check_entity_nearby(agent, 'EXPLORER', n_min=1, n_max=2) for agent in self.agents]
        metrics['check_exp_normal_plan0'] = [av_plan.check_entity_nearby(agent, 'EXPLORER', n_min=3, n_max=3) for agent in self.agents]
        # metrics['check_exp_coridor'] = [av.check_entity_nearby(agent, 'EXPLORER', n_min=2, n_max=3, env_type='coridor') for agent in self.agents]
        # metrics['check_wan_coridor'] = [av.check_entity_nearby(agent, 'WANDERER', n_min=1, n_max=2, env_type='coridor') for agent in self.agents]
        # metrics['check_exp_corner'] = [av.check_entity_nearby(agent, 'EXPLORER', n_min=2, n_max=3, env_type='corner') for agent in self.agents]
        # metrics['check_wan_corner'] = [av.check_entity_nearby(agent, 'WANDERER', n_min=1, n_max=2, env_type='corner') for agent in self.agents]
        metrics['frame_ids'] = [agent.frame_idx if isinstance(agent, DQNAgentBase) else None for agent in self.agents]
        metrics['lr_list'] = [agent.get_lr() if isinstance(agent, NNAgent) else None for agent in self.agents]

        # Add reward bonus metrics from agents that have a reward shaper
        for i, agent in enumerate(self.agents):
            if hasattr(agent, 'latest_bonus_averages') and agent.latest_bonus_averages is not None:
                for bonus_type, value in agent.latest_bonus_averages.items():
                    if bonus_type not in metrics:
                        metrics[bonus_type] = [None] * len(self.agents)
                    metrics[bonus_type][i] = value
                if 'bonused_reward' not in metrics:
                    metrics['bonused_reward'] = [None] * len(self.agents)
                metrics['bonused_reward'][i] = sum(agent.latest_bonus_averages.values())

            if hasattr(agent, 'metrics_aggregator') and agent.train:
                agent_metrics = agent.metrics_aggregator.get_metrics()
                for loss in ['policy_loss', 'value_loss', 'entropy', 'kl_div', 'loss']:
                    if loss not in metrics:
                        metrics[loss] = [None] * len(self.agents)
                    metrics[loss][i] = agent_metrics[loss]
            else:
                for loss in ['policy_loss', 'value_loss', 'entropy', 'kl_div', 'loss']:
                    if loss not in metrics:
                        metrics[loss] = [None] * len(self.agents)
                    metrics[loss][i] = getattr(agent, loss, None)

            if 'acc_weighted' not in metrics:
                metrics['acc_weighted'] = [None] * len(self.agents)
            metrics['acc_weighted'][i] = sum([
                0.4 * metrics['check_exp'][i][0],
                0.1 * metrics['check_exp_normal_plan0'][i][0],
                0.1 * metrics['check_exp_normal_plan1'][i][0],
                0.4 * metrics['check_wan'][i][0],
            ])
        return metrics

    def _log_metrics(self, step, metrics):
        for i, agent in enumerate(self.agents):
            if not agent.train:
                continue
            # Log all metrics in combined plots
            agent.writer.add_scalar('Play/Rewards', metrics['rewards'][i], step)
            agent.writer.add_scalar('Play/Winners', metrics['winner_list'][i], step)
            # agent.writer.add_scalar('Train/Policy', metrics['check_policy'][i], step)
            # agent.writer.add_scalar('Train/Epsilon', metrics['eps'][i], step)
            agent.writer.add_scalar('Check/acc_weighted', metrics['acc_weighted'][i], step)
            agent.writer.add_scalar('Check/Explorer/acc', metrics['check_exp'][i][0], step)
            agent.writer.add_scalar('Check/Wanderer/acc', metrics['check_wan'][i][0], step)
            agent.writer.add_scalar('Check/Explorer/acc_plan0', metrics['check_exp_normal_plan0'][i][0], step)
            agent.writer.add_scalar('Check/Explorer/acc_plan1', metrics['check_exp_normal_plan1'][i][0], step)
            # agent.writer.add_scalar('Check/Explorer/acc_coridor', metrics['check_exp_coridor'][i][0], step)
            # agent.writer.add_scalar('Check/Wanderer/acc_coridor', metrics['check_wan_coridor'][i][0], step)
            # agent.writer.add_scalar('Check/Explorer/acc_corner', metrics['check_exp_corner'][i][0], step)
            # agent.writer.add_scalar('Check/Wanderer/acc_corner', metrics['check_wan_corner'][i][0], step)
            # agent.writer.add_scalar('Check/Explorer/std', metrics['check_exp_normal'][i][1], step)
            # agent.writer.add_scalar('Check/Wanderer/std', metrics['check_wan_normal'][i][1], step)
            if metrics['frame_ids'][i] is not None:
                agent.writer.add_scalar('Check/frame_id', metrics['frame_ids'][i], step)
            if metrics['lr_list'][i] is not None:
                agent.writer.add_scalar('Train/lr', metrics['lr_list'][i], step)
            agent.writer.add_scalar('Check/Explorer/top_a', metrics['check_exp'][i][2], step)
            agent.writer.add_scalar('Check/Wanderer/top_a', metrics['check_wan'][i][2], step)
            # agent.writer.add_scalar('Check/Explorer/max', metrics['check_exp_normal'][i][3], step)
            # agent.writer.add_scalar('Check/Wanderer/max', metrics['check_wan_normal'][i][3], step)
            # agent.writer.add_scalar('Check/Explorer/mean', metrics['check_exp_normal'][i][4], step)
            # agent.writer.add_scalar('Check/Wanderer/mean', metrics['check_wan_normal'][i][4], step)
            
            
            # Log reward bonus metrics
            bonus_types = [
                'original_reward', 'no_move_bonus', 'wanderers_nearby_bonus', 'explorers_nearby_bonus',
                'shelters_nearby_bonus', 'others_sanity_loss_bonus', 'plan_bonus',
                'light_bonus', 'yell_bonus', 'bonused_reward',
            ]
            for bonus_type in bonus_types:
                if bonus_type in metrics and metrics[bonus_type][i] is not None:
                    agent.writer.add_scalar(f'Rewards/{bonus_type}', metrics[bonus_type][i], step)
            
            for loss in ['policy_loss', 'value_loss', 'entropy', 'kl_div', 'loss']:
                if metrics[loss][i] is not None:
                    agent.writer.add_scalar(f'Train/{loss}', metrics[loss][i], step)

    def close(self):
        """Close resources used by the trainer"""
        for i, agent in enumerate(self.agents):
            if hasattr(agent, 'writer'):
                agent.writer.close()
