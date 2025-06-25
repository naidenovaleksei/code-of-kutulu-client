import numpy as np

from src.envs.agents.nn_agent import(
    NNAgent,
    GAMMA,
    LEARNING_RATE,
)
from src.envs.buffers import (
    Experience,
    BaseStateEncoder,
)


class EpisodeBuffer:
    def __init__(self, state_encoder: BaseStateEncoder, need_aug=False):
        self.buffer = []
        self.state_encoder = state_encoder
        self.need_aug = need_aug
        
    def start_episode(self):
        self.buffer = []

    def append(self, experience):
        self.buffer.append(experience)
        
    def end_episode(self):
        assert len(self.buffer) > 0
        if self.need_aug:
            states, actions, rewards, _, _, _ = zip(*[
                self.rotation_augment(exp) for exp in self.buffer
            ])
        else:
            states, actions, rewards, _, _, _ = zip(*self.buffer)
        return states, actions, rewards

    def encode_states(self, states):
        return self.state_encoder.encode_states(states)
    
    def rotation_augment(self, exp: Experience):
        clockwise_dir = np.random.randint(0, 4)
        return Experience(
            self.state_encoder.state_rotation_augment(exp.state, clockwise_dir),
            self.state_encoder.action_rotation_augment(exp.action, clockwise_dir),
            exp.reward,
            exp.done,
            None,
            None,
        )


class ActorAgent(NNAgent):
    def __init__(self, state_type, action_space_n,
                 model, state_encoder,
                 lr=LEARNING_RATE,
                 gamma=GAMMA,
                 train=False, verbose=False, need_aug=False, **kw):
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
            episode_buffer=EpisodeBuffer(state_encoder, need_aug),
            **kw,
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
    
    def append_observation(self, player_id, reward, game_over, env_idx=None):
        if not self.train:
            return
        if reward is not None:
            state, action, observation = self.state_actions
            exp = Experience(state, action, reward, game_over, None, observation)
            self.episode_buffer.append(exp)
        else:
            assert game_over

    def train_step(self):
        if self.train:
            self._train_model()
            # Start a new episode
            self.episode_buffer.start_episode()
            self.episode_idx += 1
