import collections

import numpy as np
import torch
import torch.nn as nn
import copy
import pickle as pkl

from src.envs.agents.nn_agent import(
    NNAgent,
    GAMMA,
    LEARNING_RATE,
    EPSILON_START,
    EPSILON_FINAL,
    EPSILON_DECAY_LAST_FRAME,
)
from src.envs.models.dqn_model import DQN

BATCH_SIZE = 32
REPLAY_SIZE = 10000
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000


def parse_dir(encoded_dir):
    if encoded_dir is None:
        return [0, 0, 0, 0, 1]
    result = [0, 0, 0, 0, 0]
    for _dir in encoded_dir:
        result[_dir] = 1
    return result

def parse_dist(encoded_dist):
    if encoded_dist is None:
        return [-1]
    return [encoded_dist]


Experience = collections.namedtuple('Experience', field_names=[
    'state',
    'action',
    'reward',
    'done',
    'new_state',
    'observation',
])


class ExperienceBuffer:
    def __init__(self, capacity, need_aug=False):
        self.buffer = collections.deque(maxlen=capacity)
        self.need_aug = need_aug

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def encode_states(self, states):
        data = dict(
            closest_explorer_dir=[parse_dir(state[0]) for state in states],
            closest_explorer_dist=[parse_dist(state[1]) for state in states],
            closest_wanderer_dir=[parse_dir(state[2]) for state in states],
            closest_wanderer_dist=[parse_dist(state[3]) for state in states],
        )
        data = {k: torch.tensor(v) for k,v in data.items()}
        return data

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states, _ = zip(*[self.buffer[idx] for idx in indices])
        if self.need_aug:
            states = list(states)
            next_states = list(next_states)
            actions = list(actions)
            assert len(next_states) == batch_size
            for i in range(batch_size):
                # [+1, +2, +3]
                dir_shift = np.random.randint(1, 4)
                states[i] = self._aug_rotation_state(states[i], dir_shift)
                next_states[i] = self._aug_rotation_state(next_states[i], dir_shift)
                actions[i] = self._aug_rotation_dir(actions[i], dir_shift, ignore_errors=True)
        states = self.encode_states(states)
        next_states = self.encode_states(next_states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(np.array(rewards, dtype=np.float32))
        dones = torch.BoolTensor(np.array(dones))
        return states, actions, rewards, dones, next_states

    def save_buffer(self, fname):
        with open(fname, "wb") as f:
            pkl.dump(self.buffer, f)
    
    def load_buffer(self, fname):
        with open(fname, "rb") as f:
            self.buffer = pkl.load(f)
    
    def _aug_rotation_state(self, state, dir_shift):
        return [
            self._aug_rotation_entity(e, dir_shift)
            for e in state
        ]

    def _aug_rotation_entity(self, e, dir_shift):
        new_e = dict(e)
        if e['dir'] is not None:
            new_e['dir'] = tuple(
                self._aug_rotation_dir(d, dir_shift) for d in e['dir']
            )
        theta = np.pi / 2 * dir_shift
        new_e['rel_x'] = int(np.round(e['rel_x'] * np.cos(theta) - e['rel_y'] * np.sin(theta)))
        new_e['rel_y'] = int(np.round(e['rel_x'] * np.sin(theta) + e['rel_y'] * np.cos(theta)))
        return new_e

    def _aug_rotation_dir(self, _dir, dir_shift, ignore_errors=False):
        if _dir >= 4:
            if ignore_errors or _dir == 4:
                return _dir
            raise ValueError(f'wrong dir: {_dir}')
        return (_dir + dir_shift) % 4


class SumTree:
    """
    A binary sum tree data structure for efficient sampling based on priorities.
    
    The leaf nodes contain the priorities, and the internal nodes contain the sum of their children.
    This allows for efficient sampling proportional to priority in O(log n) time.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        # Tree structure: 2*capacity - 1 nodes in total (capacity leaf nodes + capacity-1 internal nodes)
        self.tree = np.zeros(2 * capacity - 1)
        # Data storage
        self.data = np.zeros(capacity, dtype=object)
        # Current position for cyclic buffer
        self.position = 0
        # Current size (number of filled positions)
        self.size = 0
    
    def _propagate(self, idx, change):
        """Propagate priority change up the tree"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx, s):
        """Find sample based on priority value s"""
        left = 2 * idx + 1
        right = left + 1
        
        # If we're at a leaf node
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self):
        """Return the total priority"""
        return self.tree[0]
    
    def add(self, priority, data):
        """Add a new sample with given priority"""
        # Index in the tree array where the priority will be stored
        idx = self.position + self.capacity - 1
        
        # Store data
        self.data[self.position] = data
        
        # Update tree with new priority
        self.update(idx, priority)
        
        # Update position for cyclic buffer
        self.position = (self.position + 1) % self.capacity
        
        # Update size
        self.size = min(self.size + 1, self.capacity)
    
    def update(self, idx, priority):
        """Update priority at given index"""
        # Calculate the change in priority
        change = priority - self.tree[idx]
        
        # Update the priority
        self.tree[idx] = priority
        
        # Propagate the change up the tree
        self._propagate(idx, change)
    
    def get(self, s):
        """Get sample based on a value s in range [0, total_priority)"""
        idx = self._retrieve(0, s)
        
        # Map tree index to data index
        data_idx = idx - self.capacity + 1
        
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedExperienceBuffer(ExperienceBuffer):
    """
    Prioritized Experience Replay buffer.
    
    Implements sampling based on TD error priorities using a sum tree data structure.
    Also applies importance sampling weights to correct for the bias introduced by non-uniform sampling.
    """
    def __init__(self, capacity, need_aug=False, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=1e-6):
        """
        Initialize the prioritized replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
            need_aug: Whether to apply data augmentation
            alpha: How much prioritization to use (0 = uniform sampling, 1 = full prioritization)
            beta: Importance sampling weight exponent (0 = no correction, 1 = full correction)
            beta_increment: Amount to increase beta each time sample is called
            epsilon: Small constant to add to priorities to ensure non-zero sampling probability
        """
        # Don't call the parent constructor to avoid creating the deque
        self.need_aug = need_aug
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.max_priority = 1.0
        
        # Use a sum tree for efficient priority-based sampling
        self.sum_tree = SumTree(capacity)
    
    def __len__(self):
        """Return the current size of the buffer"""
        return self.sum_tree.size
    
    def append(self, experience):
        """Add a new experience to the buffer with max priority"""
        # New experiences are added with maximum priority to ensure they are sampled at least once
        priority = (self.max_priority ** self.alpha)
        self.sum_tree.add(priority, experience)
    
    def sample(self, batch_size):
        """
        Sample a batch of experiences based on their priorities.
        
        Returns:
            Tuple of (states, actions, rewards, dones, next_states, indices, weights)
            where indices and weights are used for updating priorities
        """
        indices = []
        experiences = []
        weights = np.zeros(batch_size, dtype=np.float32)
        
        # Calculate the priority segment
        total_priority = self.sum_tree.total()
        segment = total_priority / batch_size
        
        # Increase beta over time to reduce the bias correction as learning progresses
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Calculate the max weight for normalization
        min_prob = np.min(self.sum_tree.tree[-self.sum_tree.capacity:]) / total_priority
        max_weight = (min_prob * batch_size) ** (-self.beta)
        
        for i in range(batch_size):
            # Sample uniformly from each segment
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            
            # Retrieve sample from the sum tree
            idx, priority, experience = self.sum_tree.get(s)
            
            # Calculate sampling probability
            sampling_prob = priority / total_priority
            
            # Calculate importance sampling weight
            weight = (sampling_prob * batch_size) ** (-self.beta)
            # Normalize weights to be between 0 and 1
            weights[i] = weight / max_weight
            
            indices.append(idx)
            experiences.append(experience)
        
        # Extract components from experiences
        states, actions, rewards, dones, next_states, _ = zip(*experiences)
        
        # Apply data augmentation if needed
        if self.need_aug:
            states = list(states)
            next_states = list(next_states)
            actions = list(actions)
            assert len(next_states) == batch_size
            for i in range(batch_size):
                # [+1, +2, +3]
                dir_shift = np.random.randint(1, 4)
                states[i] = self._aug_rotation_state(states[i], dir_shift)
                next_states[i] = self._aug_rotation_state(next_states[i], dir_shift)
                actions[i] = self._aug_rotation_dir(actions[i], dir_shift, ignore_errors=True)
        
        # Encode states for the neural network
        states = self.encode_states(states)
        next_states = self.encode_states(next_states)
        
        # Convert to tensors
        actions = torch.tensor(actions)
        rewards = torch.tensor(np.array(rewards, dtype=np.float32))
        dones = torch.BoolTensor(np.array(dones))
        weights = torch.tensor(weights, dtype=torch.float32)
        
        return states, actions, rewards, dones, next_states, indices, weights
    
    def update_priorities(self, indices, priorities):
        """
        Update priorities for sampled experiences.
        
        Args:
            indices: Indices of the experiences in the sum tree
            priorities: New priorities based on TD errors
        """
        for idx, priority in zip(indices, priorities):
            # Add a small epsilon to ensure non-zero sampling probability
            priority = priority + self.epsilon
            
            # Update max priority for new experiences
            self.max_priority = max(self.max_priority, priority)
            
            # Apply alpha exponent to control the amount of prioritization
            priority = priority ** self.alpha
            
            # Update the priority in the sum tree
            self.sum_tree.update(idx, priority)
    
    def save_buffer(self, fname):
        """Save the buffer to a file"""
        # Convert sum tree to a format that can be pickled
        data_to_save = {
            'data': [self.sum_tree.data[i] for i in range(self.sum_tree.size)],
            'priorities': [self.sum_tree.tree[i + self.sum_tree.capacity - 1] for i in range(self.sum_tree.size)],
            'alpha': self.alpha,
            'beta': self.beta,
            'max_priority': self.max_priority,
            'need_aug': self.need_aug
        }
        with open(fname, "wb") as f:
            pkl.dump(data_to_save, f)
    
    def load_buffer(self, fname):
        """Load the buffer from a file"""
        with open(fname, "rb") as f:
            data = pkl.load(f)
        
        # Restore parameters
        self.alpha = data['alpha']
        self.beta = data['beta']
        self.max_priority = data['max_priority']
        self.need_aug = data['need_aug']
        
        # Restore sum tree
        self.sum_tree = SumTree(self.sum_tree.capacity)
        for experience, priority in zip(data['data'], data['priorities']):
            self.sum_tree.add(priority, experience)


class DQNAgent(NNAgent):
    def __init__(self, state_type, action_space_n,
                 lr=LEARNING_RATE, replay_size=REPLAY_SIZE,
                 replay_start_size=REPLAY_START_SIZE,
                 batch_size=BATCH_SIZE,
                 need_aug=False,
                 sync_target_frames=SYNC_TARGET_FRAMES,
                 epsilon_start=EPSILON_START, epsilon_final=EPSILON_FINAL,
                 epsilon_decay_last_frame=EPSILON_DECAY_LAST_FRAME,
                 gamma=GAMMA,
                 model=None, model_params={}, episode_buffer=None,
                 train=False, verbose=False,
                 prioritized_replay=False, alpha=0.6, beta=0.4):
        if model is None:
            model = DQN(action_space_n, **model_params)
        if episode_buffer is None:
            if prioritized_replay:
                episode_buffer = PrioritizedExperienceBuffer(replay_size, need_aug, alpha, beta)
            else:
                episode_buffer = ExperienceBuffer(replay_size, need_aug)
        super(DQNAgent, self).__init__(
            state_type=state_type,
            action_space_n=action_space_n,
            lr=lr,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_final=epsilon_final,
            epsilon_decay_last_frame=epsilon_decay_last_frame,
            train=train,
            verbose=verbose,
            model=model,
            episode_buffer=episode_buffer,
        )
        self.replay_start_size = replay_start_size
        self.sync_target_frames = sync_target_frames
        self.batch_size = batch_size
        self.prioritized_replay = prioritized_replay

        self.tgt_net = copy.deepcopy(self.model)
        self.criterion = nn.MSELoss(reduction='none')  # Use 'none' to get per-sample losses for priority updates

    def generate_random_step(self, actions_masked, player_mask):
        ps = np.ma.array(np.ones(self.action_space_n), mask=player_mask).filled(0)
        if ps.sum() == 0:
            return np.random.randint(self.action_space_n)
        return np.random.choice(np.arange(self.action_space_n), p=ps / ps.sum())

    def train_step(self, reward, game_over, new_state):
        if reward is None or not self.train:
            return

        state, action, observation = self.state_actions
        if new_state is None:
            new_state = state
        exp = Experience(state, action, reward, game_over, new_state, observation)
        self.episode_buffer.append(exp)

        if len(self.episode_buffer) >= self.replay_start_size:
            self._train_model()

    def _train_model(self):
        if self.frame_idx % self.sync_target_frames == 0:
            self.tgt_net.load_state_dict(self.model.state_dict())

        self.model.train()
        self.optimizer.zero_grad()
        
        # Sample from buffer - different behavior for prioritized vs. regular buffer
        if self.prioritized_replay:
            states, actions, rewards, dones, next_states, indices, weights = self.episode_buffer.sample(self.batch_size)
            # Convert weights to tensor if not already
            weights = weights.to(actions.device)
        else:
            states, actions, rewards, dones, next_states = self.episode_buffer.sample(self.batch_size)
            weights = None
        
        # Get current state-action values
        state_action_values = self.model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        
        # Get next state values
        next_state_values = self.tgt_net(next_states).max(1)[0]
        next_state_values[dones] = 0.0
        next_state_values = next_state_values.detach()

        # Calculate expected state-action values
        expected_state_action_values = next_state_values * self.gamma + rewards
        
        # Calculate loss - per sample if using prioritized replay
        losses = self.criterion(state_action_values, expected_state_action_values)
        
        # Apply importance sampling weights if using prioritized replay
        if weights is not None:
            losses = losses * weights
        
        # Calculate mean loss for backpropagation
        loss = losses.mean()
        
        # Backpropagate
        loss.backward()
        self.optimizer.step()
        
        # Update priorities if using prioritized replay
        if self.prioritized_replay:
            # Get absolute TD errors as priorities
            priorities = losses.detach().cpu().numpy()
            self.episode_buffer.update_priorities(indices, priorities)
        
        self.last_loss = loss.item()

        if self.verbose:
            print(f"Loss: {loss.item():.4f}")
