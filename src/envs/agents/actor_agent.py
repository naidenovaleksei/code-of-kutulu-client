import numpy as np

from src.envs.agents.nn_agent import(
    NNAgent,
    GAMMA,
    LEARNING_RATE,
)
from src.envs.buffers import (
    Experience,
)


class EpisodeBuffer:
    def __init__(self, state_encoder):
        self.buffer = []
        self.state_encoder = state_encoder
        
    def start_episode(self):
        self.buffer = []

    def append(self, experience):
        self.buffer.append(experience)
        
    def end_episode(self):
        assert len(self.buffer) > 0
        states, actions, rewards, _, _, _ = zip(*self.buffer)
        return states, actions, rewards

    def encode_states(self, states):
        return self.state_encoder.encode_states(states)


class ActorAgent(NNAgent):
    def __init__(self, state_type, action_space_n,
                 model, state_encoder,
                 lr=LEARNING_RATE,
                 gamma=GAMMA,
                 train=False, verbose=False):
        super(ActorAgent, self).__init__(
            state_type=state_type,
            action_space_n=action_space_n,
            lr=lr,
            gamma=gamma,
            epsilon_start=1,
            epsilon_final=1,
            epsilon_decay_last_frame=1,
            epsilon_reset=None,
            epsilon_reset_coef=None,
            train=train,
            verbose=verbose,
            model=model,
            episode_buffer=EpisodeBuffer(state_encoder),
        )
        self.episode_idx = 0
        self.episode_buffer.start_episode()

    def get_eps(self):
        return 1

    def generate_random_step(self, actions_masked, player_mask):
        ps = actions_masked.filled(0)
        if ps.sum() == 0:
            return np.random.randint(self.action_space_n)
        return np.random.choice(np.arange(self.action_space_n), p=ps / ps.sum())

    def train_step(self, reward, game_over, new_state):
        if not self.train:
            return
        assert reward is not None

        state, action, observation = self.state_actions
        exp = Experience(state, action, reward, game_over, None, observation)
        self.episode_buffer.append(exp)

        # If the episode has ended, train on it
        if game_over or reward is None:
            self._train_model()
            # Start a new episode
            self.episode_buffer.start_episode()
            self.episode_idx += 1
