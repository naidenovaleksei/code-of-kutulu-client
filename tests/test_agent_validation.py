import pytest
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.envs.agent_validator import AgentValidator
from experiments.run_experiment import get_agent_info
from src.envs.agents.agent_factory import get_agent
from src.game.template import EXTENDED_KUTULU_ACTIONS


def get_agent_by_params(competitor_type, competitor_config, new_experiment, legacy_encoder):
    """Load an agent from experiment configuration.
    
    Args:
        competitor_type: Type of competitor ('ppo', 'qdn_conv', etc.)
        competitor_config: Experiment ID or config path
        new_experiment: Whether this is a new-style experiment
        legacy_encoder: Whether to use legacy encoder
        
    Returns:
        Loaded agent instance
    """
    agent_info = get_agent_info(competitor_config, new_experiment=new_experiment, output_dir='../output')
    agent_info['checkpoint_dir'] = agent_info['checkpoint_dir'].replace(
        "/home/kutulu/projects", "/Users/aleksei/projects"
    )
    agent_info['legacy_encoder'] = legacy_encoder
    
    agent = get_agent(agent_info)
    agent.train = True
    return agent


@pytest.fixture
def agent_validator():
    """Create an AgentValidator instance for testing."""
    return AgentValidator(EXTENDED_KUTULU_ACTIONS)


@pytest.fixture
def ppo_agent():
    """Load the PPO agent from the notebook."""
    competitor_type = 'ppo'
    competitor_config = '05abe073de06428e896fcd880c9f3eac'
    new_experiment = True
    legacy_encoder = True
    
    agent = get_agent_by_params(competitor_type, competitor_config, new_experiment, legacy_encoder)
    return agent


@pytest.fixture
def qdn_conv_agent():
    """Load the QDN Conv agent from the notebook."""
    competitor_type = 'qdn_conv'
    competitor_config = '20250622-045641'
    new_experiment = False
    legacy_encoder = False
    
    agent = get_agent_by_params(competitor_type, competitor_config, new_experiment, legacy_encoder)
    return agent


class TestPPOAgentValidation:
    """Test suite for validating PPO agent behavior.
    
    Expected values from notebook validation:
    - WANDERER check: (1.0, 0.0, 1, 0.0, 0.0)
    - SLASHER check: (0.0, 0.0, 1, 0.0, 0.0)
    - EXPLORER check: (1.0, 0.0, 4, 0.0, 0.0)
    """
    
    def test_ppo_wanderer_avoidance(self, agent_validator, ppo_agent):
        """Test that PPO agent properly avoids wanderers.
        
        Expected: (1.0, 0.0, 1, 0.0, 0.0)
        - mean_acc: 1.0 (perfect avoidance)
        - mean_std: 0.0
        - top_action: 1
        - max_output: 0.0
        - mean_output: 0.0
        """
        mean_acc, mean_std, top_action, max_output, mean_output = agent_validator.check_entity_nearby(
            ppo_agent, 'WANDERER', n_min=2, n_max=3, verbose=False
        )
        
        assert mean_acc == 1.0, f"Expected mean_acc=1.0, got {mean_acc}"
        assert mean_std == 0.0, f"Expected mean_std=0.0, got {mean_std}"
        assert top_action == 1, f"Expected top_action=1, got {top_action}"
        assert max_output == 0.0, f"Expected max_output=0.0, got {max_output}"
        assert mean_output == 0.0, f"Expected mean_output=0.0, got {mean_output}"
    
    def test_ppo_slasher_avoidance(self, agent_validator, ppo_agent):
        """Test that PPO agent properly avoids slashers.
        
        Expected: (0.0, 0.0, 1, 0.0, 0.0)
        - mean_acc: 0.0 (note: this indicates the agent doesn't avoid slashers as expected)
        - mean_std: 0.0
        - top_action: 1
        - max_output: 0.0
        - mean_output: 0.0
        """
        mean_acc, mean_std, top_action, max_output, mean_output = agent_validator.check_entity_nearby(
            ppo_agent, 'SLASHER', n_min=2, n_max=3, verbose=False
        )
        
        assert mean_acc == 0.0, f"Expected mean_acc=0.0, got {mean_acc}"
        assert mean_std == 0.0, f"Expected mean_std=0.0, got {mean_std}"
        assert top_action == 1, f"Expected top_action=1, got {top_action}"
        assert max_output == 0.0, f"Expected max_output=0.0, got {max_output}"
        assert mean_output == 0.0, f"Expected mean_output=0.0, got {mean_output}"
    
    def test_ppo_explorer_seeking(self, agent_validator, ppo_agent):
        """Test that PPO agent properly seeks explorers.
        
        Expected: (1.0, 0.0, 4, 0.0, 0.0)
        - mean_acc: 1.0 (perfect seeking)
        - mean_std: 0.0
        - top_action: 4
        - max_output: 0.0
        - mean_output: 0.0
        """
        mean_acc, mean_std, top_action, max_output, mean_output = agent_validator.check_entity_nearby(
            ppo_agent, 'EXPLORER', n_min=2, n_max=3, verbose=False
        )
        
        assert mean_acc == 1.0, f"Expected mean_acc=1.0, got {mean_acc}"
        assert mean_std == 0.0, f"Expected mean_std=0.0, got {mean_std}"
        assert top_action == 4, f"Expected top_action=4, got {top_action}"
        assert max_output == 0.0, f"Expected max_output=0.0, got {max_output}"
        assert mean_output == 0.0, f"Expected mean_output=0.0, got {mean_output}"


class TestQDNConvAgentValidation:
    """Test suite for validating QDN Conv agent behavior.
    
    Expected values from notebook validation:
    - WANDERER check: (1.0, 0.0, 1, 0.0, 0.0)
    - SLASHER check: (0.875, 0.0, 1, 0.0, 0.0)
    - EXPLORER check: (1.0, 0.0, 4, 0.0, 0.0)
    """
    
    def test_qdn_wanderer_avoidance(self, agent_validator, qdn_conv_agent):
        """Test that QDN Conv agent properly avoids wanderers.
        
        Expected: (1.0, 0.0, 1, 0.0, 0.0)
        - mean_acc: 1.0 (perfect avoidance)
        - mean_std: 0.0
        - top_action: 1
        - max_output: 0.0
        - mean_output: 0.0
        """
        mean_acc, mean_std, top_action, max_output, mean_output = agent_validator.check_entity_nearby(
            qdn_conv_agent, 'WANDERER', n_min=2, n_max=3, verbose=False
        )
        
        assert mean_acc == 1.0, f"Expected mean_acc=1.0, got {mean_acc}"
        assert mean_std == 0.0, f"Expected mean_std=0.0, got {mean_std}"
        assert top_action == 1, f"Expected top_action=1, got {top_action}"
        assert max_output == 0.0, f"Expected max_output=0.0, got {max_output}"
        assert mean_output == 0.0, f"Expected mean_output=0.0, got {mean_output}"
    
    def test_qdn_slasher_avoidance(self, agent_validator, qdn_conv_agent):
        """Test that QDN Conv agent properly avoids slashers.
        
        Expected: (0.875, 0.0, 1, 0.0, 0.0)
        - mean_acc: 0.875 (87.5% correct avoidance)
        - mean_std: 0.0
        - top_action: 1
        - max_output: 0.0
        - mean_output: 0.0
        """
        mean_acc, mean_std, top_action, max_output, mean_output = agent_validator.check_entity_nearby(
            qdn_conv_agent, 'SLASHER', n_min=2, n_max=3, verbose=False
        )
        
        assert mean_acc == 0.875, f"Expected mean_acc=0.875, got {mean_acc}"
        assert mean_std == 0.0, f"Expected mean_std=0.0, got {mean_std}"
        assert top_action == 1, f"Expected top_action=1, got {top_action}"
        assert max_output == 0.0, f"Expected max_output=0.0, got {max_output}"
        assert mean_output == 0.0, f"Expected mean_output=0.0, got {mean_output}"
    
    def test_qdn_explorer_seeking(self, agent_validator, qdn_conv_agent):
        """Test that QDN Conv agent properly seeks explorers.
        
        Expected: (1.0, 0.0, 4, 0.0, 0.0)
        - mean_acc: 1.0 (perfect seeking)
        - mean_std: 0.0
        - top_action: 4
        - max_output: 0.0
        - mean_output: 0.0
        """
        mean_acc, mean_std, top_action, max_output, mean_output = agent_validator.check_entity_nearby(
            qdn_conv_agent, 'EXPLORER', n_min=2, n_max=3, verbose=False
        )
        
        assert mean_acc == 1.0, f"Expected mean_acc=1.0, got {mean_acc}"
        assert mean_std == 0.0, f"Expected mean_std=0.0, got {mean_std}"
        assert top_action == 4, f"Expected top_action=4, got {top_action}"
        assert max_output == 0.0, f"Expected max_output=0.0, got {max_output}"
        assert mean_output == 0.0, f"Expected mean_output=0.0, got {mean_output}"


class TestComparativeValidation:
    """Compare the two agents' performance."""
    
    def test_both_agents_quality_comparison(self, agent_validator, ppo_agent, qdn_conv_agent):
        """Compare quality metrics between PPO and QDN Conv agents.
        
        PPO agent expected results:
        - WANDERER: 1.0, SLASHER: 0.0, EXPLORER: 1.0
        
        QDN Conv agent expected results:
        - WANDERER: 1.0, SLASHER: 0.875, EXPLORER: 1.0
        """
        # Test PPO agent
        ppo_wanderer, _, _, _, _ = agent_validator.check_entity_nearby(
            ppo_agent, 'WANDERER', n_min=2, n_max=3, verbose=False
        )
        ppo_slasher, _, _, _, _ = agent_validator.check_entity_nearby(
            ppo_agent, 'SLASHER', n_min=2, n_max=3, verbose=False
        )
        ppo_explorer, _, _, _, _ = agent_validator.check_entity_nearby(
            ppo_agent, 'EXPLORER', n_min=2, n_max=3, verbose=False
        )
        
        # Test QDN Conv agent
        qdn_wanderer, _, _, _, _ = agent_validator.check_entity_nearby(
            qdn_conv_agent, 'WANDERER', n_min=2, n_max=3, verbose=False
        )
        qdn_slasher, _, _, _, _ = agent_validator.check_entity_nearby(
            qdn_conv_agent, 'SLASHER', n_min=2, n_max=3, verbose=False
        )
        qdn_explorer, _, _, _, _ = agent_validator.check_entity_nearby(
            qdn_conv_agent, 'EXPLORER', n_min=2, n_max=3, verbose=False
        )
        
        # Verify expected values
        assert ppo_wanderer == 1.0
        assert ppo_slasher == 0.0
        assert ppo_explorer == 1.0
        
        assert qdn_wanderer == 1.0
        assert qdn_slasher == 0.875
        assert qdn_explorer == 1.0
        
        # QDN Conv agent should have better slasher avoidance
        assert qdn_slasher > ppo_slasher, \
            "QDN Conv agent should have better slasher avoidance than PPO agent"
