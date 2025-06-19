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
                 epsilon_start, epsilon_final, epsilon_decay_last_frame,
                 epsilon_reset, epsilon_reset_coef,
                 episode_buffer,
                 train, verbose=False, checkpoint_dir=None, drop_layers=None,
                 optimizer='adam', scheduler_params=None):
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
        self.epsilon_reset_coef = epsilon_reset_coef
        self.gamma = gamma
        self.verbose = verbose
        self.episode_buffer = episode_buffer
        self.last_loss = np.inf
        self.frame_idx = 0
        self.model = model
        self.lr = lr
        if optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer == 'adamw':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        else:
            raise ValueError(f'wrong optimizer: {optimizer}')
        if scheduler_params is None:
            self.scheduler = None
        elif scheduler_params['type'] == 'cosine':
            T_max = scheduler_params['T_max']
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max)
        else:
            raise ValueError(f'wrong scheduler_params: {scheduler_params}')
        
        
        if checkpoint_dir is not None:
            self.load_agent(checkpoint_dir, drop_layers)

    def get_eps(self):
        return max(
            self.epsilon_final,
            self.eps,
        )

    def get_lr(self):
        if self.scheduler is None:
            return self.lr
        else:
            return self.scheduler.get_last_lr()[0]

    def generate_state_and_step(self, player_id, need_update=True):
        if need_update:
            self.frame_idx += 1
            self._update_eps()

        valid_actions = self.get_valid_actions(player_id)
        player_mask = ~np.array(valid_actions)
        player_mask = player_mask[:self.action_space_n]

        state = self.get_state(player_id)
        data = self.episode_buffer.encode_states([state])
        
        # Move data to device if the agent has device attribute (for GPU support)
        if hasattr(self, 'device') and hasattr(self, '_move_states_to_device'):
            data = self._move_states_to_device(data)
        
        self.model.eval()
        model_output = self.model.get_policy(data)[0].detach().cpu().numpy()
        actions_masked = np.ma.array(model_output, mask=player_mask)

        self.output_std = actions_masked.max() - (actions_masked.sum() - actions_masked.max()) / (np.sum(valid_actions) - 1)
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

    def load_agent(self, checkpoint_dir, drop_layers=None):
        # Load to CPU first, then move to device if needed
        device = getattr(self, 'device', 'cpu')
        map_location = 'cpu' if device == 'cpu' else None
        
        if drop_layers:
            weights = torch.load(f"{checkpoint_dir}/model.pt", map_location=map_location)
            for layer in drop_layers:
                del weights[layer]
            self.model.load_state_dict(weights, strict=False)
        else:
            weights = torch.load(f"{checkpoint_dir}/model.pt", map_location=map_location)
            self.model.load_state_dict(weights)
        
        # Move model to device after loading
        if hasattr(self, 'device'):
            self.model = self.model.to(self.device)
            # Also update target network if it exists
            if hasattr(self, 'tgt_net'):
                self.tgt_net = self.tgt_net.to(self.device)

    def _update_eps(self):
        self.eps -= self.epsilon_start / self.epsilon_decay_last_frame
        if self.epsilon_reset is not None:
            if self.frame_idx >= self.epsilon_reset:
                self.eps = min(self.epsilon_start, self.eps * 2)
                self.epsilon_reset *= self.epsilon_reset_coef

    def generate_random_step(self, actions_masked, player_mask):
        raise NotImplementedError
    
    def _train_model(self):
        raise NotImplementedError

    def train_step(self, reward, game_over, new_state):
        raise NotImplementedError
