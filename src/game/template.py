import sys
import math
import numpy as np
import scipy.special as sp
import heapq

import pickle as pkl
import zlib
import base64


data1 = b'data1data1data1'
data2 = b'data2data2data2'


MOVE_REL_POS = {
    'UP': (0,-1),
    'RIGHT': (1,0),
    'DOWN': (0,1),
    'LEFT': (-1,0),
    'WAIT': (0,0)
}
REL_POSITIONS = (
    MOVE_REL_POS['UP'],
    MOVE_REL_POS['RIGHT'],
    MOVE_REL_POS['DOWN'],
    MOVE_REL_POS['LEFT'],
    MOVE_REL_POS['WAIT'],
)
REL_SHIFT = {pos: abs(pos[0]) + abs(pos[1]) for pos in REL_POSITIONS}

MAX_EXPLORER_DIST = 5
MAX_WANDERER_DIST = 5

DEFAULT_KUTULU_ACTIONS = [
    'UP',
    'RIGHT',
    'DOWN',
    'LEFT',
    'WAIT',
]

CELL_EMPTY = '.'
CELL_WALL = '#'
CELL_SPAWN = 'w'

class UnreachedPositionError(Exception):
    pass

def distance(point1, point2):
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

def add(point1, point2):
    return (point1[0] + point2[0], point1[1] + point2[1])

def find_path(start_point, finish_point, lines):
    choices = ((0,1), (1,0), (0,-1), (-1,0))
    g_dict = {start_point: 0}
    f_dict = {start_point: distance(start_point, finish_point)}
    common_candidates = [(f_dict[start_point], start_point)]
    close_set = set()
    pred_point = {}

    while len(common_candidates):
        point = heapq.heappop(common_candidates)[1]
        if point == finish_point:
            path = []
            while point != start_point:
                path.append(point)
                point = pred_point[point]
            return path

        close_set.add(point)
        for choise in choices:
            candidate = add(point, choise)
            x,y = candidate
            # if lines[y][x] not in (CELL_EMPTY, CELL_SPAWN):
            if lines[y][x] == CELL_WALL:
                continue
            g = g_dict[point] + 1
            h = distance(candidate, finish_point)
            f = g + h
            if candidate in close_set and g >= g_dict.get(candidate, 0):
                continue
            pred_point[candidate] = point
            if g < g_dict.get(candidate, 0):
                g_dict[candidate] = g
            if candidate in [x[1] for x in common_candidates]:
                continue
            if candidate in f_dict:
                continue
            g_dict[candidate] = g
            f_dict[candidate] = f
            heapq.heappush(common_candidates, (f, candidate))
    raise UnreachedPositionError(f'cannot reach {finish_point} from {start_point}')

def parse_desc(line):
    unit_kind, unit_id, unit_x, unit_y, eparam0, eparam1, eparam2 = line.split()
    unit_id = int(unit_id)
    unit_x = int(unit_x)
    unit_y = int(unit_y)
    unit_life = int(eparam0)
    return {
        "kind": unit_kind,
        "id": unit_id,
        "x": unit_x,
        "y": unit_y,
        "life": unit_life,
        "wandering": int(eparam1),
        "param0": int(eparam0),
        "param1": int(eparam1),
        "param2": int(eparam2),
    }

def get_distances(entities, player_pos, lines, find_path_func=find_path):
    distances = {}
    for rel_pos in REL_POSITIONS:
        pos = (rel_pos[0] + player_pos[0], rel_pos[1] + player_pos[1])
        if not 0 < pos[0] < len(lines[0]) - 1:
            continue
        if not 0 < pos[1] < len(lines) - 1:
            continue
        if lines[pos[1]][pos[0]] == CELL_WALL:
            continue
        for e in entities:
            entity_pos = (e['x'], e['y'])
            if entity_pos == player_pos:
                return {MOVE_REL_POS['WAIT']: 0}
            try:
                path = find_path_func(pos, entity_pos, lines)
                if len(path):
                    assert path[-1] != pos
                    assert path[0] == entity_pos
                distances[rel_pos] = min(distances.get(rel_pos, 1000), len(path))
            except UnreachedPositionError as e:
                pass
    return distances

def get_min_direction_and_distance(distances):
    if len(distances) == 0:
        return None, None
    d_min = min(distances.values())
    key = tuple(
        i
        for i, rel_pos in enumerate(REL_POSITIONS)
        if distances.get(rel_pos) == d_min
    )
    return key, d_min + REL_SHIFT[REL_POSITIONS[key[0]]]


def get_state(player_pos, entities, lines, get_distances_func=get_distances):
    explorers = [unit for unit in entities[1:] if unit["kind"] == "EXPLORER"]
    wanderers = [unit for unit in entities[1:] if unit["kind"] == "WANDERER" and unit["wandering"] == 1]
    
    explorer_distances = get_distances_func(explorers, player_pos, lines)
    wanderers_distances = get_distances_func(wanderers, player_pos, lines)
    closest_explorer_dir, closest_explorer_dist = get_min_direction_and_distance(explorer_distances)
    closest_wanderer_dir, closest_wanderer_dist = get_min_direction_and_distance(wanderers_distances)

    if closest_explorer_dist is not None:
        closest_explorer_dist = min(closest_explorer_dist, MAX_EXPLORER_DIST)
    if closest_wanderer_dist is not None:
        closest_wanderer_dist = min(closest_wanderer_dist, MAX_WANDERER_DIST)
    return (
        closest_explorer_dir,
        closest_explorer_dist,
        closest_wanderer_dir,
        closest_wanderer_dist,
    )


def get_state_bronze(player_pos, entities, lines, get_distances_func=get_distances):
    explorers = [unit for unit in entities[1:] if unit["kind"] == "EXPLORER"]
    wanderers = [
        unit for unit in entities[1:]
        if (unit["kind"] == "WANDERER" and unit["wandering"] == 1) or
           (unit["kind"] == "SLASHER")
    ]

    explorer_distances = get_distances_func(explorers, player_pos, lines)
    wanderers_distances = get_distances_func(wanderers, player_pos, lines)
    closest_explorer_dir, closest_explorer_dist = get_min_direction_and_distance(explorer_distances)
    closest_wanderer_dir, closest_wanderer_dist = get_min_direction_and_distance(wanderers_distances)

    if closest_explorer_dist is not None:
        closest_explorer_dist = min(closest_explorer_dist, MAX_EXPLORER_DIST)
    if closest_wanderer_dist is not None:
        closest_wanderer_dist = min(closest_wanderer_dist, MAX_WANDERER_DIST)
    return (
        closest_explorer_dir,
        closest_explorer_dist,
        closest_wanderer_dir,
        closest_wanderer_dist,
    )


def get_state_ext(player_pos, entities, lines, get_distances_func=get_distances):
    ENTITY_FIELDS = ['kind', 'rel_x', 'rel_y', 'param0', 'param1', 'param2', 'dist', 'dir', 'raw_dist', 'on_los']
    
    entities_features = []
    for e in entities[1:]:
        distances = get_distances_func([e], player_pos, lines)
        _dir, _dist = get_min_direction_and_distance(distances)
        e['rel_x'] = e['x'] - player_pos[0]
        e['rel_y'] = e['y'] - player_pos[1]
        e['dist'] = _dist
        e['dir'] = _dir
        e['raw_dist'] = abs(e['rel_x']) + abs(e['rel_y'])
        e['on_los'] = (e['rel_x'] == 0 or e['rel_y'] == 0) and e['raw_dist'] == e['dist']
        e = {k: v for k,v in e.items() if k in ENTITY_FIELDS}
        entities_features.append(e)
    return entities_features


def getActionGreedyMasked(state, Q, action_space_n, mask):
    if state not in Q:
        actions_idx = np.arange(action_space_n)[~mask]
        if len(actions_idx) == 0:
            return np.random.randint(action_space_n)
        return np.random.choice(actions_idx)
    assert len(Q[state]) == action_space_n
    a = np.ma.array(Q[state], mask=mask)
    a_star = a.argmax()
    return a_star

def getActionGreedyMasked2(state, Q, action_space_n, mask):
    if not isinstance(state, tuple) or state not in Q:
        ps = np.ma.array(np.ones(action_space_n), mask=mask).filled(0)
        if ps.sum() == 0:
            return np.random.randint(action_space_n)
        return np.random.choice(np.arange(action_space_n), p=ps / ps.sum())
    assert len(Q[state]) == action_space_n
    a = np.ma.array(Q[state], mask=mask)
    a_star = a.argmax()
    return a_star

def calculate_output_np(data, weights, num_classes=5):
    data = {k: np.array(v) for k,v in data.items()}
    kind_embs = weights['kind_embs.weight']
    features_linear = weights['features_linear.weight']
    features_linear_b = weights['features_linear.bias']
    dir_linear = weights['dir_linear.weight']
    dir_linear_b = weights['dir_linear.bias']
    entity_linear = weights['entity_linear.weight']
    entity_linear_b = weights['entity_linear.bias']
    entity_impact = weights['entity_impact.weight']
    entity_impact_b = weights['entity_impact.bias']
    out_linear = weights['out_linear.weight']
    out_linear_b = weights['out_linear.bias']
    
    assert data['entity_dir'].shape[-1] == num_classes
    x_kind_embs = kind_embs[data['entity_kind']]
    
    entity_features = data['entity_features']
    x_features = entity_features @ features_linear.T + features_linear_b

    entity_dir = data['entity_dir']
    x_dir = entity_dir @ dir_linear.T + dir_linear_b
    entities_mask = (entity_dir > 0).max(axis=-1, keepdims=True)

    x_entitity = np.concatenate((x_kind_embs, x_features, x_dir), axis=-1)
    x = x_entitity @ entity_linear.T + entity_linear_b

    entities_mask[entities_mask == 0] = -100000
    entity_weights = (x_entitity @ entity_impact.T + entity_impact_b) * entities_mask
    entity_weights = sp.softmax(entity_weights, axis=1)

    # [batch_size, inner_dim, entity_dim]
    x_transposed = x.transpose(0, 2, 1)
    # [batch_size, inner_dim, num_classes]
    x = (x_transposed @ entity_weights)
    # [batch_size, num_classes, inner_dim]
    x = x.transpose(0, 2, 1)
    # [batch_size, num_classes]
    output = (x @ out_linear.T + out_linear_b).squeeze(-1)

    return output

def get_valid_action_mask_by_coords(x, y, info, can_wait=True):
    # 'UP',
    # 'RIGHT',
    # 'DOWN',
    # 'LEFT',
    # 'WAIT',
    return [
        y > 0 and info['lines'][y - 1][x] != CELL_WALL,
        x < info['width'] - 1 and info['lines'][y][x + 1] != CELL_WALL,
        y < info['height'] - 1 and info['lines'][y + 1][x] != CELL_WALL,
        x > 0 and info['lines'][y][x - 1] != CELL_WALL,
        can_wait
    ]

def simple_strategy(entities, new_Q, info):
    player_pos = (entities[0]['x'], entities[0]['y'])
    state = get_state_bronze(player_pos, entities, info['lines'])
    player_mask = ~np.array(get_valid_action_mask_by_coords(player_pos[0], player_pos[1], info))
    action_id = getActionGreedyMasked2(state, new_Q, len(DEFAULT_KUTULU_ACTIONS), player_mask)
    
    rel_pos = MOVE_REL_POS[DEFAULT_KUTULU_ACTIONS[action_id]]
    
    next_cell = (player_pos[0] + rel_pos[0], player_pos[1] + rel_pos[1])

    return f"MOVE {next_cell[0]} {next_cell[1]}"

def parse_info():
    width = int(input())
    height = int(input())
    lines = []
    for i in range(height):
        line = input()
        lines.append( line )
        print(line, file=sys.stderr)
    sanity_loss_lonely, sanity_loss_group, wanderer_spawn_time, wanderer_life_time = [int(i) for i in input().split()]
    return {
        'width': width,
        'height': height,
        'lines': lines,
        'sanity_loss_lonely': sanity_loss_lonely,
        'sanity_loss_group': sanity_loss_group,
        'wanderer_spawn_time': wanderer_spawn_time,
        'wanderer_life_time': wanderer_life_time,
    }

def main():
    vals = pkl.loads(zlib.decompress(base64.b64decode(data1)))
    keys = pkl.loads(zlib.decompress(base64.b64decode(data2)))
    new_Q = dict(zip(keys, vals))
    
    info = parse_info()

    
    # game loop
    while True:
        entity_count = int(input())  # the first given entity corresponds to your explorer
        entities = []
        for i in range(entity_count):
            entities.append(parse_desc(input()))
        
        step = simple_strategy(entities, new_Q, info)
        print(step)

if __name__ == '__main__':
    main()
