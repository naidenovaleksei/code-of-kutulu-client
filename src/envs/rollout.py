import numpy as np

TEST_STATES = [
    ((4,), 0, None, None),
    ((0,), 1, None, None),
    ((1,), 1, None, None),
    ((2,), 1, None, None),
    ((3,), 1, None, None),
    ((0,), 2, None, None),
    ((1,), 2, None, None),
    ((2,), 2, None, None),
    ((3,), 2, None, None),
    ((0,), 3, None, None),
    ((1,), 3, None, None),
    ((2,), 3, None, None),
    ((3,), 3, None, None),
    ((0,), 4, None, None),
    ((1,), 4, None, None),
    ((2,), 4, None, None),
    ((3,), 4, None, None),
]


def viz_map(env_map, entities):
    curr_map = [list(line) for line in env_map]
    for e in entities:
        curr_map[e['y']][e['x']] = e['type'][0]
        if e['type'] == 'EXPLORER':
            curr_map[e['y']][e['x']] = str(e['id'])
    curr_map = [''.join(line) for line in curr_map]
    for line in curr_map:
        print(line)

def rollout(env, seed, verbose=False, can_wait=True):
    info = env.reset(seed)
    rewards = []
    game_over = False
    while not game_over:
        action = env.sample_valid_action(can_wait)
#         action = [4, 4, 4, 4]
        entities, reward, game_over, info = env.step(action)
        if verbose:
            viz_map(env.map, entities)
        rewards.append(reward)
    return rewards

def check_policy(Q):
    result_list = []
    for state in TEST_STATES:
        _dir = state[0][0]
        result = np.argmax(Q.get(state, [0])) == _dir
        result_list.append(result)
    return np.mean(result_list)
