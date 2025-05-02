import numpy as np
from src.envs.kutulu_world import KutuluWorldEnv, DEFAULT_KUTULU_ACTIONS
from src.envs.agent import Agent, CrossEntropyAgent
from src.envs.kutulu_observer import (
    KutuluClosestObserver,
    KutuluClosestBronzeObserver
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
    def __init__(self, num_experiments, agents_info, can_wait, league_level, reward_for_win, mazes=BRONZE_MAZES, shuffle=True):
        self.num_experiments = num_experiments
        self.action_space_n = len(DEFAULT_KUTULU_ACTIONS) - 1 + int(can_wait)
        self.mazes = mazes
        self.league_level = league_level
        self.players_count = len(agents_info)
        self.reward_for_win = reward_for_win
        self.can_wait = can_wait
        self.shuffle = shuffle
        
        self.env = None
        self.observers = None

        self.agents = []
        for agent_info in agents_info:
            agent_info['action_space_n'] = self.action_space_n
            if 'strategy' in agent_info:
                agent = Agent(**agent_info)
            else:
                assert agent_info['cross_entropy']
                agent_info = dict(agent_info)
                del agent_info['cross_entropy']
                agent = CrossEntropyAgent(**agent_info)
            self.agents.append(agent)
            
        self.agent_map = np.arange(len(self.agents))
    
    def reset_env(self):
        maze_name = np.random.choice(self.mazes)
        self.env = KutuluWorldEnv(
            server_host='localhost:8080',
            maze_name=maze_name,
            league_level=self.league_level,
            players_count=self.players_count,
            reward_for_win=self.reward_for_win
        )
        observation, info = self.env.reset()
        
        closest_observer = KutuluClosestObserver(self.env)
        closest_bronze_observer = KutuluClosestBronzeObserver(self.env)
        self.observers = []
        for agent in self.agents:
            if agent.state_type == 'closest':
                self.observers.append(closest_observer)
            elif agent.state_type == 'closest_bronze':
                self.observers.append(closest_bronze_observer)
            else:
                raise ValueError(f'unknown state_type: {agent.state_type}')

        return self.env
    
    def play_rollout(self):
        env = self.reset_env()
        rollout_rewards = []
        game_over = False
        if self.shuffle:
            np.random.shuffle(self.agent_map)
        while not game_over:
            action = env.sample_valid_action(self.can_wait)

            for env_player in env.active_players():
                player_id = env_player['id']
                agent_id = self.agent_map[player_id]
                state, At = self.agents[agent_id].generate_state_and_step(
                    self.observers[agent_id], player_id)
                action[player_id] = At

            entities, rewards, game_over, info = env.step(action)
            rollout_rewards.append(rewards)

            for player_id, reward in enumerate(rewards):
                agent_id = self.agent_map[player_id]
                agent = self.agents[agent_id]
                if agent.train:
                    try:
                        new_state = self.observers[agent_id].get_state(
                            player_id, self.agents[agent_id].state_type
                        )
                        agent.train_step(reward, game_over, new_state)
                    except StopIteration:
                        pass
        rollout_rewards = np.array(rollout_rewards, dtype=float)[:,np.argsort(self.agent_map)]
        return rollout_rewards
