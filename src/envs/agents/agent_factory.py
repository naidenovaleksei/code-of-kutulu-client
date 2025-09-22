from copy import deepcopy
from src.envs.agents.qlearning_agent import QlearningAgent
from src.envs.agents.cross_entropy_agent import CrossEntropyAgent
from src.envs.agents.dqn_agent import DQNAgent
from src.envs.agents.dqn_agent_ext import DQNAgentExt
from src.envs.agents.dqn_agent_by_kind import DQNAgentByKind
from src.envs.agents.reinforce_agent import REINFORCEAgent
from src.envs.agents.a2c_agent import A2CAgent
from src.envs.agents.dqn_agent_conv import DQNAgentConv
from src.envs.agents.ppo_agent import PPOAgent
from src.envs.agents.rule_based_agent import EpsilonConstAgent
from src.envs.agents.stupid_wrapper_agent import StupidAgentWrapper
from src.envs.agents.dummy_agent import DummyAgent
from src.envs.strategy import RandomStrategy


def get_agent(agent_info):
    agent_info = deepcopy(agent_info)
    _type = agent_info.pop('type')
    wrapper_params = None
    if 'wrapper_params' in agent_info:
        wrapper_params = agent_info.pop('wrapper_params')
    if _type == 'qlearning':
        if agent_info['strategy'] == 'random':
            agent_info['strategy'] = RandomStrategy()
        else:
            raise ValueError(f'wrong strategy: {agent_info["strategy"]}')
        agent = QlearningAgent(**agent_info)
    elif _type == 'epsilon_wait':
        agent = EpsilonConstAgent(**agent_info)
    elif _type == 'dummy':
        agent = DummyAgent(**agent_info)
    elif _type == 'cross_entropy':
        agent = CrossEntropyAgent(**agent_info)
    elif _type == 'qdn':
        agent = DQNAgent(**agent_info)
    elif _type == 'qdn_ext':
        agent = DQNAgentExt(**agent_info)
    elif _type == 'reinforce':
        agent = REINFORCEAgent(**agent_info)
    elif _type == 'a2c':
        agent = A2CAgent(**agent_info)
    elif _type == 'ppo':
        agent = PPOAgent(**agent_info)
    elif _type == 'qdn_by_kind':
        agent = DQNAgentByKind(**agent_info)
    elif _type == 'qdn_conv':
        agent = DQNAgentConv(**agent_info)
    else:
        raise ValueError(f'unknown kind: {_type}')
    if wrapper_params is not None:
        agent = StupidAgentWrapper(**wrapper_params, agent=agent)
    return agent
