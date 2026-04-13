"""
Agent Metrics Calculator

This module provides the Metrics class for calculating various performance metrics
for agents, including behavior validation against different entity types.
"""
import numpy as np

from src.envs.agent_validator import AgentValidator
from src.envs.trainer import Trainer
from src.game.template import EXTENDED_KUTULU_ACTIONS

import os
os.environ['KUTULU_ARTIFACT_OUTPUT']='../../output'

class Metrics:
    """
    Calculate performance metrics for agents.
    
    This class evaluates agent behavior by testing their responses to various
    game scenarios involving EXPLORER, WANDERER, and SLASHER entities.
    """
    
    @classmethod
    def load_default_competitors(cls):
        """
        Load the default competitor agents for challenge mode.
        
        This loads two pre-trained agents:
        - 'ppo': A PPO agent trained with legacy encoder
        - 'qdn_conv': A DQN convolutional agent
        
        Returns:
            Dictionary of {competitor_name: agent} for challenge matches
            
        Raises:
            ImportError: If required modules are not available
            FileNotFoundError: If competitor artifacts are not found
        """
        from experiments.run_experiment import get_agent_info
        from src.envs.agents.agent_factory import get_agent
        
        competitors = {}
        for competitor_type, competitor_config, new_experiment, legacy_encoder in [
            ('ppo', '05abe073de06428e896fcd880c9f3eac', True, True),
            ('qdn_conv', '20250622-045641', False, False),
        ]:
            agent_info = get_agent_info(competitor_config, new_experiment=new_experiment)
            agent_info['legacy_encoder'] = legacy_encoder
            competitors[competitor_type] = get_agent(agent_info)
        
        return competitors
    
    def __init__(self, use_challenge=False, competitors=None):
        """
        Initialize the Metrics calculator.
        
        Args:
            use_challenge: If True, enable challenge mode. If competitors is None,
                          will attempt to load default competitors.
            competitors: Optional dict of {name: agent} for challenge matches.
                        If None and use_challenge=True, loads default competitors.
        """
        self.agent_validator = AgentValidator(EXTENDED_KUTULU_ACTIONS)
        self.agent_validator_plan = AgentValidator(
            EXTENDED_KUTULU_ACTIONS, 
            player_params=(100, 1, 0)
        )
        
        # Load competitors if challenge mode is enabled
        if use_challenge:
            if competitors is None:
                try:
                    self.competitors = self.load_default_competitors()
                except (ImportError, FileNotFoundError, Exception) as e:
                    # Gracefully handle missing competitors
                    import warnings
                    warnings.warn(
                        f"Could not load default competitors: {e}. "
                        "Challenge mode will be disabled."
                    )
                    self.competitors = None
            else:
                self.competitors = competitors
        else:
            self.competitors = None

    def _play_round(self, agents, league_level, num_envs, seed):
        """
        Play a single round with the given agents.
        
        Args:
            agents: List of 4 agents to play
            league_level: Game difficulty level
            num_envs: Number of parallel environments
            seed: Random seed for reproducibility
            
        Returns:
            Array of scores (survival steps) for each agent
        """
        assert len(agents) == 4, "Exactly 4 agents required"
        
        trainer = Trainer(
            num_experiments=1,
            agents_info=None,
            agents=agents,
            shuffle=True,
            league_level=league_level,
            seed=seed,
            silent=True,
            num_envs=num_envs,
            only_train=False,
            use_tqdm=False,
        )
        result = trainer.play_rollout(only_eval=True)
        
        # Calculate scores as the number of steps each agent survived
        scores = np.array([
            np.argwhere(~np.isnan(result[:, i])).max().item() 
            for i in range(4)
        ])
        return scores

    def _calculate_metrics(self, agent, use_challenge=True, n_exps=20, rollouts_seed=17, verbose=False):
        """
        Calculate comprehensive metrics for an agent.
        
        This method evaluates:
        - Agent behavior against EXPLORER entities
        - Agent behavior against WANDERER entities  
        - Agent behavior against SLASHER entities
        - Weighted accuracy scores
        - Optional challenge matches against competitors
        
        Args:
            agent: The agent to evaluate
            use_challenge: Whether to run challenge matches (requires competitors)
            
        Returns:
            Dictionary of metric names to values
        """
        metrics = {}
        av = self.agent_validator
        av_plan = self.agent_validator_plan
        
        # Check behavior against different entity types
        metrics['check_exp'] = av.check_entity_nearby(
            agent, 'EXPLORER', n_min=2, n_max=3, 
            env_types=('normal', 'coridor', 'corner'),
            verbose=verbose,
        )
        metrics['check_wan'] = av.check_entity_nearby(
            agent, 'WANDERER', n_min=1, n_max=2, 
            env_types=('normal', 'coridor', 'corner'),
            verbose=verbose,
        )
        metrics['check_slsh'] = av.check_entity_nearby(
            agent, 'SLASHER', n_min=2, n_max=3, 
            env_types=('normal',),
            verbose=verbose,
        )
        
        # Check behavior with PLAN effect active
        metrics['check_exp_normal_plan1'] = av_plan.check_entity_nearby(
            agent, 'EXPLORER', n_min=1, n_max=2,
            verbose=verbose,
        )
        metrics['check_exp_normal_plan0'] = av_plan.check_entity_nearby(
            agent, 'EXPLORER', n_min=3, n_max=3,
            verbose=verbose,
        )
        
        # Calculate weighted accuracy scores
        metrics['acc_weighted'] = sum([
            0.4 * metrics['check_exp'][0],
            0.1 * metrics['check_exp_normal_plan0'][0],
            0.1 * metrics['check_exp_normal_plan1'][0],
            0.4 * metrics['check_wan'][0],
        ])
        
        metrics['acc_weighted_full'] = sum([
            0.3 * metrics['check_exp'][0],      # 0-1
            0.2 * metrics['check_wan'][0],      # 0-1
            0.3 * metrics['check_slsh'][0],     # 0-1
            0.05 / 4 * metrics['check_exp'][2], # 1-4
            0.05 / 4 * metrics['check_wan'][2], # 1-4
            0.05 * metrics['check_exp_normal_plan0'][0], # 0-1
            0.05 * metrics['check_exp_normal_plan1'][0], # 0-1
        ])
        
        # Run challenge matches if competitors are available
        if use_challenge and self.competitors is not None:
            rollouts_rng = np.random.RandomState(rollouts_seed)
            for competitor_type, competitor in self.competitors.items():
                agents = [agent, competitor, competitor, competitor]
                n_wins = 0
                for _ in range(n_exps):
                    rollout_seed = rollouts_rng.randint(999999)
                    scores = self._play_round(agents, 4, 1, rollout_seed)
                    if scores[0] == np.max(scores):
                        n_wins += 1
                metrics[f'winner_score_{competitor_type}'] = n_wins / n_exps
        
        return metrics
