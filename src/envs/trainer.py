import numpy as np
from src.envs.kutulu_world import (
    KutuluWorldEnv,
)
from src.envs.agents.qlearning_agent import QlearningAgent
from src.envs.agents.cross_entropy_agent import CrossEntropyAgent
from src.envs.agents.dqn_agent import DQNAgent
from src.envs.agents.dqn_agent_ext import DQNAgentExt
from src.envs.agents.reinforce_agent import REINFORCEAgent
from src.envs.kutulu_observer import (
    KutuluClosestObserver,
    KutuluClosestExtObserver,
)

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
                 env_kwargs=None, mazes=BRONZE_MAZES, shuffle=True):
        self.num_experiments = num_experiments
        self.mazes = mazes
        self.league_level = league_level
        self.players_count = len(agents_info)
        self.env_kwargs = env_kwargs or {}
        self.shuffle = shuffle
        self.actions = actions
        
        self.env = None
        self.observers = None

        self.agents = []
        for agent_info in agents_info:
            if 'strategy' in agent_info:
                agent = QlearningAgent(**agent_info)
            elif 'cross_entropy' in agent_info:
                agent_info = dict(agent_info)
                del agent_info['cross_entropy']
                agent = CrossEntropyAgent(**agent_info)
            elif 'qdn' in agent_info:
                agent_info = dict(agent_info)
                del agent_info['qdn']
                agent = DQNAgent(**agent_info)
            elif 'qdn_ext' in agent_info:
                agent_info = dict(agent_info)
                del agent_info['qdn_ext']
                agent = DQNAgentExt(**agent_info)
            elif 'reinforce' in agent_info:
                agent_info = dict(agent_info)
                del agent_info['reinforce']
                agent = REINFORCEAgent(**agent_info)
            else:
                raise ValueError(f'unknown kind: {agent_info}')
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
                        try:
                            new_state = agent.get_state(player_id)
                            agent.train_step(reward, game_over, new_state)
                        except StopIteration:
                            pass
        rollout_rewards = np.array(rollout_rewards, dtype=float)[:,np.argsort(self.agent_map)]
        return rollout_rewards
