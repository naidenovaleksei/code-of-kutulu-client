def viz_map(env_map, entities):
    curr_map = [list(line) for line in env_map]
    for e in entities:
        curr_map[e['y']][e['x']] = e['type'][0]
        if e['type'] == 'EXPLORER':
            curr_map[e['y']][e['x']] = str(e['id'])
    curr_map = [''.join(line) for line in curr_map]
    for line in curr_map:
        print(line)

def rollout(env, seed, verbose=False):
    info = env.reset(seed)
    rewards = []
    game_over = False
    while not game_over:
        action = env.sample_valid_action()
#         action = [4, 4, 4, 4]
        entities, reward, game_over, info = env.step(action)
        if verbose:
            viz_map(env.map, entities)
        rewards.append(reward)
    return rewards
