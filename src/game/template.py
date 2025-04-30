import sys
import math
import numpy as np
import heapq

import pickle as pkl
import zlib
import base64


data1 = b'eNoBLgLR/YADY251bXB5LmNvcmUubXVsdGlhcnJheQpfcmVjb25zdHJ1Y3QKcQBjbnVtcHkKbmRhcnJheQpxAUsAhXECQwFicQOHcQRScQUoSwFLCksFhnEGY251bXB5CmR0eXBlCnEHWAIAAABmOHEIiYiHcQlScQooSwNYAQAAADxxC05OTkr/////Sv////9LAHRxDGKJQpABAABAS329cVbfP0fg1b2kfNw/oWI/7FVY3z9Tgm3UslHfP1vmzRmEUt8/XMHPyxY22z8QVsl3HSvbP13ShvILN9s/Tfh6+Io32z/BQorQOsjXP+IdwCitpNK/wKOY2upPzj9Gpo0UUaTkvyaxFRPGXd2/RccQ/IFQzj+kOKtquIrkPznmaOfWwu4/Jy4mapnt8b/QQven3DXyvzJESYHjXPS/2olKCE2ccT/wHgeKY8X8v2zc0mwItvu/lI2N7WHS/r8XEI1BRB3+vzRBFMbk3s0/TKv6YscrwD9EzqfyZ7y+PwAO1AWnwyE/XFF8cBRU5L+KQ4P62DH4vx6fsuiLEvm/f4C8cfux+b9jDm9HS7n3vyhvmfFnEPe/GZqtckun6L8QVBJz8NruP1ejIlFB++2/ASpPcUc+zj/gcX6q+rXnv6pdpxOfMfq/Cg9sK3MD+L9KcC8u5lv4v9pVCitMkvi/hf/8x33n+L+/qInee00FwK5MufqRDgTAErmdELLAA8D/f/sxCyEFwM+YK1yGhgTAcQ10cQ5iLuI2CWo='
data2 = b'eNo1y00OgjAQxfF2BlQUBD9Qj9AVF3kJN2DpAV7SA7gpXls6jZnV7+U/H13oQoAkevh5jhSDwmVUAT6xLtgFOPiV+xIeNiY2BUcLT5CM1tBBttOVZ9SRvT1DvhxQlf0Cjbxae/u397yNtj2g0MQnJPL1nn7JzikP'


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
    assert False

def parse_desc(line):
    unit_kind, unit_id, unit_x, unit_y, unit_life, eparam1, _ = line.split()
    unit_id = int(unit_id)
    unit_x = int(unit_x)
    unit_y = int(unit_y)
    unit_life = int(unit_life)
    return {
        "kind": unit_kind,
        "id": unit_id,
        "x": unit_x,
        "y": unit_y,
        "life": unit_life,
        "wandering": int(eparam1),
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
            path = find_path_func(pos, entity_pos, lines)
            if len(path):
                assert path[-1] != pos
                assert path[0] == entity_pos
            distances[rel_pos] = min(distances.get(rel_pos, 1000), len(path))
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
    state = get_state(player_pos, entities, info['lines'])
    player_mask = ~np.array(get_valid_action_mask_by_coords(player_pos[0], player_pos[1], info))
    action_id = getActionGreedyMasked(state, new_Q, len(DEFAULT_KUTULU_ACTIONS), player_mask)
    
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
