class BaseAgent:
    def __init__(self):
        pass
    def generate_state_and_step(self, observer, player_id):
        raise NotImplementedError

    def train_step(self, reward, game_over, new_state):
        raise NotImplementedError

    def check_policy(self):
        raise NotImplementedError