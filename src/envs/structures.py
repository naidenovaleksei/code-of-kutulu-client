import numpy as np


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
