import pytest
import numpy as np
from src.envs.strategy import LazyGreedyStrategy

@pytest.mark.parametrize("state, mask, expected", [
    [((0,), 0, None, None), [False, False, False, False, False], 4],
    [((0,), 0, None, None), [False, False, True, True, True], 1],
    [((0,), 1, None, None), [True, True, False, True, True], 2],
])
def test_find_path_sample(state, mask, expected):
    pi = LazyGreedyStrategy()
    pi.Q = {((0,), 0, None, None): np.array([0, 1, 2, 3, 4])}
    result = pi.getActionGreedyMasked(state, 5, np.array(mask))
    assert result == expected
