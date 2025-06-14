import collections

import numpy as np
import torch
import pickle as pkl

from src.envs.structures import SumTree

Experience = collections.namedtuple('Experience', field_names=[
    'state',
    'action',
    'reward',
    'done',
    'new_state',
    'observation',
])


class BaseStateEncoder:
    def encode_states(self, states, return_tensors=True):
        raise NotImplementedError

    def action_rotation_augment(self, action, clockwise_dir):
        clockwise_dir = clockwise_dir % 4
        assert clockwise_dir >= 0 and clockwise_dir < 4
        if action >= 4:
            return action
        return (action + clockwise_dir) % 4
    
    def state_rotation_augment(self, state, clockwise_dir):
        raise NotImplementedError


class ExperienceBuffer:
    def __init__(self, state_encoder: BaseStateEncoder, capacity, need_aug=False):
        self.buffer = collections.deque(maxlen=capacity)
        self.need_aug = need_aug
        self.state_encoder = state_encoder

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def encode_states(self, states):
        return self.state_encoder.encode_states(states)
    
    def _sample_experiences(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        if self.need_aug:
            states, actions, rewards, dones, next_states, _ = zip(*[
                self.rotation_augment(self.buffer[idx]) for idx in indices
            ])
        else:
            states, actions, rewards, dones, next_states, _ = zip(*[
                self.buffer[idx] for idx in indices
            ])
        return states, actions, rewards, dones, next_states

    def sample(self, batch_size):
        zipped_episodes = self._sample_experiences(batch_size)
        zipped_episodes = self._encode_experiences(
            *zipped_episodes
        )
        return zipped_episodes
    
    def _encode_experiences(self, states, actions, rewards, dones, next_states):
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
    
    def rotation_augment(self, exp: Experience):
        clockwise_dir = np.random.randint(0, 4)
        return Experience(
            self.state_encoder.state_rotation_augment(exp.state, clockwise_dir),
            self.state_encoder.action_rotation_augment(exp.action, clockwise_dir),
            exp.reward,
            exp.done,
            self.state_encoder.state_rotation_augment(exp.new_state, clockwise_dir),
            None,
        )


class PrioritizedExperienceBuffer(ExperienceBuffer):
    """
    Prioritized Experience Replay buffer.
    
    Implements sampling based on TD error priorities using a sum tree data structure.
    Also applies importance sampling weights to correct for the bias introduced by non-uniform sampling.
    """
    def __init__(self, state_encoder: BaseStateEncoder, capacity, need_aug=False, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=1e-6):
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
        self.state_encoder = state_encoder
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
    
    def _sample_experiences(self, batch_size):
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
            
            if self.need_aug:
                experience = self.rotation_augment(experience)
            
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
        return states, actions, rewards, dones, next_states, indices, weights

    def _encode_experiences(self, states, actions, rewards, dones, next_states,
                            indices, weights):
        states, actions, rewards, dones, next_states = super()._encode_experiences(
            states, actions, rewards, dones, next_states
        )
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