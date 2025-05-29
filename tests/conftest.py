import pytest

from src.envs.kutulu_world import KutuluWorldEnv
from src.game.template import DEFAULT_KUTULU_ACTIONS


@pytest.fixture
def mock_env():
    """Create a mock KutuluWorldEnv instance with a predefined map."""
    env = KutuluWorldEnv('', '', 1, actions=DEFAULT_KUTULU_ACTIONS)
    env.map = [
        '###########',
        '#.........#',
        '#.#.#.#.#.#',
        '#.........#',
        '#.#.#.#.#.#',
        '#.........#',
        '###########',
    ]
    env.width = len(env.map[0])
    env.height = len(env.map)
    return env
