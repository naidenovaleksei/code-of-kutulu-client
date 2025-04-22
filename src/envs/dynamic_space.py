import gym

class DynamicActionSpace(gym.Space):
    def __init__(self, env):
        super(DynamicActionSpace, self).__init__()
        self.env = env
        
    def sample(self):
        return self.env.sample_valid_action()
        
    def contains(self, x):
        mask = self.env._get_valid_action_mask()
        return 0 <= x < len(mask) and mask[x] == 1
