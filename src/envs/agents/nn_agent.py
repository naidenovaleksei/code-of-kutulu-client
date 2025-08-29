import numpy as np
import torch
import torch.optim as optim

from src.envs.agents import BaseAgent
from src.envs.agent_metric_aggregator import AgentMetricsAggregator

GAMMA = 0.99
LEARNING_RATE = 1e-4
EPSILON_START = 1.0
EPSILON_FINAL = 0.02
EPSILON_DECAY_LAST_FRAME = 10**5

DEFAULT_ACTION_MASK = [
    True,
    True,
    True,
    True,
    False,
    True,
    True,
    False,
]

class NNAgent(BaseAgent):
    def __init__(self, state_type, action_space_n,
                 model,
                 lr, gamma,
                 epsilon_start, epsilon_final, epsilon_decay_last_frame,
                 epsilon_reset, epsilon_reset_coef,
                 episode_buffer,
                 train, verbose=False, checkpoint_dir=None, drop_layers=None,
                 optimizer='adam', scheduler_params=None, explicit_random=False,
                 explicit_action_mask=DEFAULT_ACTION_MASK):
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
        self.explicit_random = explicit_random
        self.metrics_aggregator = AgentMetricsAggregator(self.verbose)
        if explicit_action_mask is not None:
            self.explicit_action_mask = np.array(explicit_action_mask)
        else:
            self.explicit_action_mask = None
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

    def generate_state_and_step(self, player_id, need_update=True,
                                return_value=False):
        if need_update:
            self.frame_idx += 1
            self._update_eps()

        valid_actions = self.get_valid_actions(player_id)
        player_mask = np.array(valid_actions)
        player_mask = player_mask[:self.action_space_n]
        if self.explicit_action_mask is not None:
            player_mask &= self.explicit_action_mask
        player_mask = ~player_mask

        state = self.get_state(player_id)
        data = self.episode_buffer.encode_states([state])

        # Move data to device if the agent has device attribute (for GPU support)
        if hasattr(self, 'device') and hasattr(self, '_move_states_to_device'):
            data = self._move_states_to_device(data)

        self.model.eval()
        with torch.no_grad():
            if return_value:
                policy, value = self.model(data)
            else:
                policy = self.model.get_policy(data)
        model_output = policy[0].detach().cpu().numpy()
        actions_masked = np.ma.array(model_output, mask=player_mask)

        self.last_action = actions_masked
        if (self.train or self.explicit_random) and np.random.random() < self.get_eps():
            action = self.generate_random_step(actions_masked, player_mask)
        else:
            action = actions_masked.argmax()

        self.state_actions = (state, action, self.get_raw_observation(player_id))
        
        output = {
            'state': state,
            'action': action.item(),
            'model_output': model_output,
            'valid_actions': valid_actions,
        }
        if return_value:
            output['value'] = value
            output['policy'] = policy
        return output

    def inference_step(self, player_id):
        valid_actions = self.get_valid_actions(player_id)
        player_mask = np.array(valid_actions)
        player_mask = player_mask[:self.action_space_n]
        if self.explicit_action_mask is not None:
            player_mask &= self.explicit_action_mask
        player_mask = ~player_mask

        state = self.get_state(player_id)
        data = self.episode_buffer.encode_states([state])

        # Move data to device if the agent has device attribute (for GPU support)
        if hasattr(self, 'device') and hasattr(self, '_move_states_to_device'):
            data = self._move_states_to_device(data)

        self.model.eval()
        with torch.no_grad():
            policy = self.model.get_policy(data)
        model_output = policy[0].detach().cpu().numpy()
        actions_masked = np.ma.array(model_output, mask=player_mask)
        action = actions_masked.argmax()

        return {
            'state': state,
            'action': action.item(),
            'model_output': model_output,
            'valid_actions': valid_actions,
        }

    def check_policy(self):
        metrics = self.metrics_aggregator.get_metrics()
        return metrics.get('loss', self.last_loss)

    def save_agent(self, checkpoint_dir):
        torch.save(self.model.state_dict(), f"{checkpoint_dir}/model.pt")

    def load_agent(self, checkpoint_dir, drop_layers=None):
        if self.verbose:
            print(f"Loading agent from '{checkpoint_dir}'")
        # Load to CPU first, then move to device if needed
        device = getattr(self, 'device', 'cpu')
        map_location = 'cpu' if device == 'cpu' else None
        
        if drop_layers:
            weights = torch.load(f"{checkpoint_dir}/model.pt", map_location=map_location)
            for layer in drop_layers:
                if layer in weights:
                    del weights[layer]
            self.model.load_state_dict(weights, strict=False)
        else:
            weights = torch.load(f"{checkpoint_dir}/model.pt", map_location=map_location)
            if 'fc1.weight' in weights and hasattr(self.model, 'fc'):
                weights['fc.weight'] = weights.pop('fc1.weight')
            if 'fc1.bias' in weights and hasattr(self.model, 'fc'):
                weights['fc.bias'] = weights.pop('fc1.bias')
            self.model.load_state_dict(weights, strict=False)
        
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

    def train_step(self):
        raise NotImplementedError
