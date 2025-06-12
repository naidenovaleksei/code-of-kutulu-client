import numpy as np
import torch
import torch.optim as optim

from src.envs.agents import BaseAgent

GAMMA = 0.99
LEARNING_RATE = 1e-4
EPSILON_START = 1.0
EPSILON_FINAL = 0.02
EPSILON_DECAY_LAST_FRAME = 10**5

class NNAgent(BaseAgent):
    def __init__(self, state_type, action_space_n,
                 model,
                 lr, gamma,
                 epsilon_start, epsilon_final, epsilon_decay_last_frame, epsilon_reset,
                 episode_buffer,
                 train, verbose=False):
        super(NNAgent, self).__init__(
            state_type,
            action_space_n,
            train,
        )
        self.epsilon_final = epsilon_final
        self.epsilon_start = epsilon_start
        self.eps = self.epsilon_start
        self.epsilon_decay_last_frame = epsilon_decay_last_frame
        self.epsilon_reset = epsilon_reset
        self.gamma = gamma
        self.verbose = verbose
        self.episode_buffer = episode_buffer
        self.last_loss = np.inf
        self.frame_idx = 0
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def get_eps(self):
        return max(
            self.epsilon_final,
            self.eps,
        )

    def generate_state_and_step(self, player_id, need_update=True):
        if need_update:
            self.frame_idx += 1
            self._update_eps()

        valid_actions = self.get_valid_actions(player_id)
        player_mask = ~np.array(valid_actions)
        player_mask = player_mask[:self.action_space_n]

        state = self.get_state(player_id)
        data = self.episode_buffer.encode_states([state])
        model_output = self.model(data)[0].detach().cpu().numpy()
        actions_masked = np.ma.array(model_output, mask=player_mask)

        self.output_std = (actions_masked / actions_masked.sum()).std()
        if self.train and np.random.random() < self.get_eps():
            action = self.generate_random_step(actions_masked, player_mask)
        else:
            action = actions_masked.argmax()

        self.state_actions = (state, action, self.get_raw_observation(player_id))
        return state, action

    def check_policy(self):
        return self.last_loss

    def save_agent(self, checkpoint_dir):
        torch.save(self.model.state_dict(), f"{checkpoint_dir}/model.pt")

    def load_agent(self, checkpoint_dir):
        self.model.load_state_dict(torch.load(f"{checkpoint_dir}/model.pt"))

    def _update_eps(self):
        self.eps -= self.epsilon_start / self.epsilon_decay_last_frame
        if self.epsilon_reset is not None:
            if self.frame_idx == self.epsilon_reset:
                self.eps = min(self.epsilon_start, self.eps * 2)
                self.epsilon_reset *= 2

    def generate_random_step(self, actions_masked, player_mask):
        raise NotImplementedError
    
    def _train_model(self):
        raise NotImplementedError

    def train_step(self, reward, game_over, new_state):
        raise NotImplementedError
        