
from src.envs.experiments import _get_agent_info

class AgentDescription:
    def __init__(self, exp_name, best_iter, agent_id, output_dir='../../../output'):
        self.exp_name = exp_name
        self.best_iter = best_iter
        self.agent_id = agent_id
        self.output_dir = output_dir

    def __eq__(self, other):
        if not isinstance(other, AgentDescription):
            return NotImplemented
        return self.exp_name == other.exp_name and self.best_iter == other.best_iter and self.agent_id == other.agent_id

    def __hash__(self):
        return hash(self.key)

    def __repr__(self):
        return str(self.key)

    @property
    def key(self):
        return (self.exp_name, self.best_iter, self.agent_id)

    # def find_agent(self):
    #     return _get_agent_info(self.exp_name, self.best_iter, self.agent_id)

    @property
    def agent_info(self):
        return _get_agent_info(self.exp_name, self.best_iter, self.agent_id, output_dir=self.output_dir)

    @classmethod
    def from_dict(cls, data):
        pass
