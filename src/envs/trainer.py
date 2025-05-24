import os
from datetime import datetime
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from src.envs.kutulu_world import KutuluWorldEnv
from src.envs.agent_validator import AgentValidator
from src.envs.agents.qlearning_agent import QlearningAgent
from src.envs.agents.cross_entropy_agent import CrossEntropyAgent
from src.envs.agents.dqn_agent import DQNAgent
from src.envs.agents.dqn_agent_ext import DQNAgentExt
from src.envs.agents.reinforce_agent import REINFORCEAgent

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

class Trainer:
    def __init__(self, num_experiments, agents_info, league_level, actions,
                 env_kwargs=None, mazes=BRONZE_MAZES, shuffle=True, log_dir='runs', exp_name=None):
        self.num_experiments = num_experiments
        self.mazes = mazes
        self.league_level = league_level
        self.players_count = len(agents_info)
        self.env_kwargs = env_kwargs or {}
        self.shuffle = shuffle
        self.actions = actions

        if exp_name is None:
            exp_name = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.log_dir = os.path.join(log_dir, exp_name)
        
        self.env = None
        self.agent_validator = AgentValidator(self.actions)

        self.agents = []
        for i, agent_info in enumerate(agents_info):
            agent_info = dict(agent_info)
            _type = agent_info.pop('type')
            if _type == 'qlearning':
                agent = QlearningAgent(**agent_info)
            elif _type == 'cross_entropy':
                agent = CrossEntropyAgent(**agent_info)
            elif _type == 'qdn':
                agent = DQNAgent(**agent_info)
            elif _type == 'qdn_ext':
                agent = DQNAgentExt(**agent_info)
            elif _type == 'reinforce':
                agent = REINFORCEAgent(**agent_info)
            else:
                raise ValueError(f'unknown kind: {_type}')
            agent_dir = f'agent{i}_{_type}'
            agent_log_dir = os.path.join(self.log_dir, agent_dir)
            agent.writer = SummaryWriter(log_dir=agent_log_dir)
            self.agents.append(agent)

        self.agent_map = np.arange(len(self.agents))
    
    def reset_env(self, seed=None):
        if seed is not None:
            maze_name = np.random.RandomState(seed).choice(self.mazes)
        else:
            maze_name = np.random.choice(self.mazes)
        self.env = KutuluWorldEnv(
            server_host='localhost:8080',
            maze_name=maze_name,
            league_level=self.league_level,
            players_count=self.players_count,
            actions=self.actions,
            **self.env_kwargs
        )
        observation, info = self.env.reset(seed=seed)

        for agent in self.agents:
            agent.set_env(self.env)

        return self.env
    
    def play_rollout(self):
        env = self.reset_env()
        rollout_rewards = []
        game_over = False
        if self.shuffle:
            np.random.shuffle(self.agent_map)
        while not game_over:
            # action = env.sample_valid_action()
            # WAIT
            action = [4 for _ in range(len(self.agents))]

            for player_id in env.active_players():
                agent_id = self.agent_map[player_id]
                state, At = self.agents[agent_id].generate_state_and_step(player_id)
                action[player_id] = At

            entities, rewards, game_over, info = env.step(action)
            rollout_rewards.append(rewards)

            for player_id, reward in enumerate(rewards):
                agent_id = self.agent_map[player_id]
                agent = self.agents[agent_id]
                if agent.train:
                    if isinstance(agent, REINFORCEAgent):
                        if env.is_game_over_for_player(player_id):
                            agent.train_step(reward, True, None)
                    else:
                        if player_id in env.active_players():
                            new_state = agent.get_state(player_id)
                            agent.train_step(reward, game_over, new_state)
        rollout_rewards = np.array(rollout_rewards, dtype=float)[:,np.argsort(self.agent_map)]
        return rollout_rewards

    def save_models(self):
        date, time = str(datetime.now()).split()
        checkpoints_dir = f'../output/{date}/{time}'
        try:
            os.mkdir(checkpoints_dir)
        except FileExistsError:
            pass
        for i,agent in enumerate(self.agents):
            checkpoint_dir = f'{checkpoints_dir}/agent{i}'
            os.mkdir(checkpoint_dir)
            agent.save_agent(checkpoint_dir)
        return checkpoints_dir

    def train(self, metrics_int=10, save_models_int=100):
        reward_list = []
        model_dir_list = []

        for i in tqdm(range(self.num_experiments)):
            rollout_rewards = self.play_rollout()
            reward_list.append(rollout_rewards)
            
            step = i + 1
            # Save models periodically
            if step % save_models_int == 0:
                model_dir = self.save_models()
                model_dir_list.append(model_dir)
                
            # Log metrics periodically
            if step % metrics_int == 0:
                total_reward_list = np.array([np.nansum(rewards, axis=0) for rewards in reward_list])
                rewards = np.mean(total_reward_list, axis=0)
                winner_list = np.argmax(total_reward_list, 1)
                winner_list = [np.sum(winner_list == i) / len(winner_list) for i, agent in enumerate(self.agents)]
                check_policy = [agent.check_policy() for agent in self.agents]
                # output_stds = [agent.get_output_std() for agent in self.agents]
                eps = [agent.get_eps() for agent in self.agents]
                av = self.agent_validator
                check_exp = [av.check_entity_nearby(agent, 'EXPLORER', n_min=2, n_max=3) for agent in self.agents]
                check_wan = [av.check_entity_nearby(agent, 'WANDERER', n_min=1, n_max=2) for agent in self.agents]
                
                for i, agent in enumerate(self.agents):
                    # Log all metrics in combined plots
                    agent.writer.add_scalar('Play/Rewards', rewards[i], step)
                    agent.writer.add_scalar('Play/Winners', winner_list[i], step)
                    agent.writer.add_scalar('Train/Policy', check_policy[i], step)
                    agent.writer.add_scalar('Train/Epsilon', eps[i], step)
                    # agent.writer.add_scalar('Train/OutputStd', output_stds[i], step)
                    agent.writer.add_scalar('Check/Explorer/acc', check_exp[i][0], step)
                    agent.writer.add_scalar('Check/Wanderer/acc', check_wan[i][0], step)
                    agent.writer.add_scalar('Check/Explorer/std', check_exp[i][1], step)
                    agent.writer.add_scalar('Check/Wanderer/std', check_wan[i][1], step)
                
        
        for i, agent in enumerate(self.agents):
            agent.writer.close()
        return reward_list, model_dir_list
        
    def close(self):
        """Close resources used by the trainer"""
        for i, agent in enumerate(self.agents):
            if hasattr(agent, 'writer'):
                agent.writer.close()