import sys
import io
import math
import numpy as np
import scipy.special as sp
from scipy.signal import correlate2d
import heapq
from collections import defaultdict
import warnings

import pickle as pkl
import zlib
import base64


data1 = b'data1data1data1'
data2 = b'data2data2data2'

mode = 'mode'
SIZE = 3


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

MAX_ENTITY_COUNT = 20
MAX_ENTITY_COUNT_BY_KIND = {
    "EXPLORER": 3,
    "WANDERER": 10,
    "SLASHER": 4,
    "EFFECT_PLAN": 2,
    "EFFECT_LIGHT": 2,
    "EFFECT_SHELTER": 2,
    "EFFECT_YELL": 1,
}


WANDERER_STATES = ["SPAWNING", "WANDERING", "STALKING", "RUSHING", "STUNNED"]
ENTITY_TOKENS = [
    "EXPLORER",
    "WANDERER_SPAWNING",
    "WANDERER_WANDERING",
    "SLASHER_SPAWNING",
    "SLASHER_WANDERING",
    "SLASHER_STALKING",
    "SLASHER_RUSHING",
    "SLASHER_STUNNED",
    "EFFECT_PLAN",
    "EFFECT_LIGHT",
    "EFFECT_SHELTER",
    "EFFECT_YELL",
]
ENTITY_TOKENS_MAP = {k: v for v, k in enumerate(ENTITY_TOKENS)}

MOVING_KUTULU_ACTIONS = [
    'UP',
    'RIGHT',
    'DOWN',
    'LEFT',
]

DEFAULT_KUTULU_ACTIONS = MOVING_KUTULU_ACTIONS + [
    'WAIT',
]

EXTENDED_KUTULU_ACTIONS = DEFAULT_KUTULU_ACTIONS + [
    'PLAN',
    'LIGHT',
    'YELL',
]

USED_ACTIONS = DEFAULT_KUTULU_ACTIONS

CELL_EMPTY = '.'
CELL_WALL = '#'
CELL_SPAWN = 'w'


MAP_MAP = {
    '#': 0,
    '.': 1,
    'w': 1,
    's': 1,
    'S': 1,
    'U': 2,
}
MAP_MAP_NEW = {
    '#': 0,
    '.': 1,
    'w': 1,
    's': 1,
    'S': 1,
    'U': 1,
}
FEATURE_ENTITY_DICT = {
    'EXPLORER': ['param0', 'param1', 'param2'],
    'WANDERER': ['param0', 'param1'],
    'SLASHER': ['param0', 'param1'],
    'EFFECT_PLAN': ['param0'],
    'EFFECT_LIGHT': ['param0'],
    'EFFECT_SHELTER': ['param0'],
    'EFFECT_YELL': ['param0']
}

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

def parse_dist_dir(e):
    encoded_dir = e['dir']
    if encoded_dir is None:
        return [0., 0., 0., 0., 0.]
    result = [0., 0., 0., 0., 0.]
    for _dir in encoded_dir:
        result[_dir] = 1
    return result

def parse_kind(e):
    kind = e['kind']
    if kind in ('WANDERER', 'SLASHER'):
        kind = f"{e['kind']}_{WANDERER_STATES[e['param1']]}"
    return ENTITY_TOKENS_MAP[kind] + 1

def parse_features(e):
    result = [
        e['param0'],
        e['param1'],
        e['param2'],
        e['rel_x'],
        e['rel_y'],
        e['dist'] if e['dist'] is not None else 100,
        e['raw_dist'],
        e['on_los'],
    ]
    result = [float(x) for x in result]
    return result

def parse_state(state, max_entity_count=MAX_ENTITY_COUNT):
    state = state[:max_entity_count]
    kind_list = [
        parse_kind(e) for e in state
    ]
    features_list = [parse_features(e) for e in state]
    dir_list = [parse_dist_dir(e) for e in state]
    for _ in range(max_entity_count - len(state)):
        kind_list.append(0)
        features_list.append([0., 0., 0., 0., 0., 0., 0., 0.])
        dir_list.append([0., 0., 0., 0., 0.])
    return kind_list, features_list, dir_list

def parse_state_by_kind(state, kinds=[
    "EXPLORER", "WANDERER", "SLASHER",
    "EFFECT_PLAN", "EFFECT_LIGHT", "EFFECT_SHELTER", "EFFECT_YELL"
    ]):
    result = {}
    for kind in kinds:
        state_by_kind = [e for e in state if e['kind'] == kind]
        kind_list, features_list, dir_list = parse_state(
            state_by_kind, MAX_ENTITY_COUNT_BY_KIND[kind])
        result[kind] = (kind_list, features_list, dir_list)
    return result

def get_all_distances(entities, player_pos, lines, find_path_func=find_path):
    all_distances = defaultdict(list)
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
                all_distances[MOVE_REL_POS['WAIT']].append(0)
            try:
                path = find_path_func(pos, entity_pos, lines)
                if len(path):
                    assert path[-1] != pos
                    assert path[0] == entity_pos
                all_distances[rel_pos].append(len(path))
            except UnreachedPositionError as e:
                pass
    return all_distances

def get_distances(entities, player_pos, lines, find_path_func=find_path):
    all_distances = get_all_distances(entities, player_pos, lines, find_path_func)
    distances = {k: min(v) for k, v in all_distances.items()}
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
    wanderers = [
        unit for unit in entities[1:]
        if (unit["kind"] == "WANDERER" and unit["param1"] == 1) or
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


def get_state_conv(player_id, entities, lines, size):
    data = {}
    curr_map = [list(line) for line in lines]
    agent_entity = None
    for e in entities:
        if e['id'] == player_id:
            agent_entity = e
            break

    assert agent_entity is not None, f"Agent with id {player_id} not found in entities"

    agent_pos = (agent_entity['x'], agent_entity['y'])
    curr_map = [[MAP_MAP['#']] * int(size * 2 + 1) for i in range(int(size * 2 + 1))]
    for rel_x in range(-size, size + 1):
        for rel_y in range(-size, size + 1):
            y = agent_pos[1] + rel_y
            x = agent_pos[0] + rel_x
            if 0 <= y and y < len(lines) and 0 <= x and x < len(lines[0]):
                curr_map[size + rel_y][size + rel_x] = MAP_MAP[lines[y][x]]
    data['map'] = curr_map
    for kind, features in FEATURE_ENTITY_DICT.items():
        for feature in features:
            curr_map = [[0] * int(size * 2 + 1) for i in range(int(size * 2 + 1))]
            if kind == 'EXPLORER':
                curr_map[size][size] = agent_entity[feature]
            for e in entities:
                if kind == 'EXPLORER' and e['id'] == agent_entity['id']:
                    continue
                rel_x = max(min(e['x'] - agent_pos[0], size), -size)
                rel_y = max(min(e['y'] - agent_pos[1], size), -size)
                if abs(rel_x) <= size and abs(rel_y) <= size and e['kind'] == kind:
                    curr_map[size + rel_y][size + rel_x] = e[feature]
            data[f'{kind}_{feature}'] = curr_map
    return data

def get_state_conv_ext(player_id, entities, lines, size):
    """
    Возвращает dict карт (окно (2*size+1)x(2*size+1) вокруг агента):
      - 'map'                         : базовая карта (через MAP_MAP)
      - 'EXPLORER_COUNT'                  : нормированная count-карта союзников (клип до 3)
      - 'WANDERER_COUNT'                 : нормированная count-карта врагов (wanderer+slasher, клип до 3)
      - 'EXPLORER_MIN_DIST'               : мин. расстояние (BFS с учётом стен) до ближайшего союзника
      - 'WANDERER_MIN_DIST'              : мин. расстояние (BFS с учётом стен) до ближайшего врага
      - 'SLASHER_LOS'                 : 1, если клетка сейчас в прямой видимости хоть одного слэшера
      - 'SLASHER_TIME_TO_LAND'        : нормированное (инвертированное) минимальное время для любого слэшера
                                         «вывести клетку на LOS и сделать рывок» с учётом его состояния (FSM)
      - а также ваши прежние каналы по FEATURE_ENTITY_DICT (без бага с клиппингом на границе)
    Предполагается наличие глобальных:
      MAP_MAP: dict[char->int] для клеток карты
      FEATURE_ENTITY_DICT: dict[kind->list[feature]]
    """
    from collections import deque
    from functools import lru_cache

    # ---------- константы состояний SLASHER ----------
    STATE_SPAWNING = 0
    STATE_WANDERING = 1
    STATE_STALKING = 2
    STATE_RUSHING  = 3
    STATE_STUNNED  = 4

    # # минимальная задержка до попытки «подвести на линию + дэш»
    # STATE_OFFSET = {
    #     STATE_SPAWNING: 6,
    #     STATE_WANDERING: 0,  # можно поставить 1, если хотите добавить «тик на вход в stalking»
    #     STATE_STALKING:  2,  # нет трекинга прогресса -> худший случай
    #     STATE_RUSHING:   0,
    #     STATE_STUNNED:   6,
    # }

    # ---------- хелперы ----------
    def in_window(ax, ay, bx, by, r):
        return abs(ax - bx) <= r and abs(ay - by) <= r

    def crop_map(full_map, agent_pos, r, pad_val, cells_map=None):
        H, W = len(full_map), len(full_map[0])
        S = 2 * r + 1
        if cells_map is not None:
            pad_val = cells_map[pad_val]
        out = [[pad_val for _ in range(S)] for __ in range(S)]
        ax, ay = agent_pos
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                x, y = ax + dx, ay + dy
                if 0 <= y < H and 0 <= x < W:
                    cell_value = full_map[y][x]
                    if cells_map is not None:
                        cell_value = cells_map[cell_value]
                    out[dy + r][dx + r] = cell_value
        return out

    def count_map(entities, agent_pos, r, clip=3):
        S = 2 * r + 1
        ax, ay = agent_pos
        mp = [[0 for _ in range(S)] for __ in range(S)]
        for e in entities:
            (x, y) = (e['x'], e['y'])
            if in_window(x, y, ax, ay, r):
                dx, dy = x - ax, y - ay
                v = mp[dy + r][dx + r] + 1
                mp[dy + r][dx + r] = v if v < clip else clip
        return mp  # ints 0..clip

    def norm01_map(int_map, denom, invert=False):
        # return int_map
        S = len(int_map)
        out = [[0.0 for _ in range(S)] for __ in range(S)]
        d = float(max(1, denom))
        for y in range(S):
            for x in range(S):
                v = int_map[y][x] / d
                if v < 0.0: v = 0.0
                if v > 1.0: v = 1.0
                out[y][x] = (1.0 - v) if invert else v
        return out

    def bfs_min_dist_in_window(lines_, sources_, r, agent_pos, max_d=None):
        """BFS по окну вокруг агента (учёт стен '#'). Возвращает int-карту min-дистанций (клип до max_d)."""
        if max_d is None:
            max_d = 2 * r + 1
        S = 2 * r + 1
        INF = 10**9
        dist = [[INF for _ in range(S)] for __ in range(S)]

        ax, ay = agent_pos
        # локальная проходимость по окну
        passable = [[True for _ in range(S)] for __ in range(S)]
        H, W = len(lines_), len(lines_[0])
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                x, y = ax + dx, ay + dy
                cell = '#'
                if 0 <= y < H and 0 <= x < W:
                    cell = lines_[y][x]
                passable[dy + r][dx + r] = (cell != '#')

        q = deque()
        for s in sources_:
            (sx, sy) = (s['x'], s['y'])
            if not in_window(sx, sy, ax, ay, r):
                continue
            lx, ly = sx - ax + r, sy - ay + r
            if passable[ly][lx]:
                dist[ly][lx] = 0
                q.append((lx, ly))

        while q:
            x, y = q.popleft()
            d = dist[y][x] + 1
            if d > max_d:
                continue
            for nx, ny in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):
                if 0 <= nx < S and 0 <= ny < S and passable[ny][nx] and dist[ny][nx] > d:
                    dist[ny][nx] = d
                    q.append((nx, ny))

        for y in range(S):
            for x in range(S):
                if dist[y][x] > max_d:
                    dist[y][x] = max_d
        return dist

    def build_wanderer_spawning_map(lines_, wanderers):
        H, W = len(lines_), len(lines_[0])
        los = [[0 for _ in range(W)] for __ in range(H)]
        for w in wanderers:
            state = w['param1']
            if state != STATE_SPAWNING:
                continue
            time_to_spawn = w['param0']
            wx, wy = w['x'], w['y']
            los[wy][wx] = max(1 / time_to_spawn, los[wy][wx])
        return los

    def get_straight_line_cells(lines_, sx, sy, tx, ty):
        H, W = len(lines_), len(lines_[0])
        cells = set([(sx, sy)])
        if tx == sx:
            dy = np.sign(ty - sy)
            y = sy + dy
            while y < H and y >= 0 and (tx, ty) not in cells and lines_[y][sx] != '#':
                cells.add((sx, y))
                y += dy
        elif ty == sy:
            dx = np.sign(tx - sx)
            x = sx + dx
            while x < W and x >= 0 and (tx, ty) not in cells and lines_[sy][x] != '#':
                cells.add((x, sy))
                x += dx
        else:
            raise ValueError("target and slasher must stand on the straight line")
        return cells

    def build_slasher_stalking_los_map(lines_, slashers, explorers):
        H, W = len(lines_), len(lines_[0])
        los = [[0 for _ in range(W)] for __ in range(H)]
        for s in slashers:
            state = s['param1']
            if state != STATE_STALKING:
                continue
            time_to_rush = s['param0']
            sx, sy = s['x'], s['y']
            target_id = s['param2']
            line_cells = None
            for e in explorers:
                if e['id'] == target_id:
                    line_cells = get_straight_line_cells(lines_, sx, sy, e['x'], e['y'])
                    break
            if line_cells is not None:
                for (x, y) in line_cells:
                    los[y][x] = max(1 / time_to_rush, los[y][x])
            else:
                warnings.warn("LOS line_cells is None")
        return los

    def build_slasher_spawning_los_map(lines_, slashers):
        H, W = len(lines_), len(lines_[0])
        los = [[0 for _ in range(W)] for __ in range(H)]
        for s in slashers:
            state = s['param1']
            if state != STATE_SPAWNING:
                continue
            time_to_spawn = s['param0']
            sx, sy = s['x'], s['y']
            line_cells = set()
            for x, y in [(sx, 0), (sx, H), (0, sy), (W, sy)]:
                line_cells |= get_straight_line_cells(lines_, sx, sy, x, y)
            if line_cells is not None:
                for (x, y) in line_cells:
                    los[y][x] = max(1 / time_to_spawn, los[y][x])
            else:
                warnings.warn("LOS line_cells is None")
        return los

    def build_slasher_wandering_los_map(lines_, slashers):
        H, W = len(lines_), len(lines_[0])
        los = [[0 for _ in range(W)] for __ in range(H)]
        for s in slashers:
            state = s['param1']
            if state != STATE_WANDERING:
                continue
            sx, sy = s['x'], s['y']
            line_cells = set()
            for x, y in [(sx, 0), (sx, H), (0, sy), (W, sy)]:
                line_cells |= get_straight_line_cells(lines_, sx, sy, x, y)
            if line_cells is not None:
                for (x, y) in line_cells:
                    los[y][x] = 1
            else:
                warnings.warn("LOS line_cells is None")
        return los

    def build_slasher_stunned_map(lines_, slashers):
        H, W = len(lines_), len(lines_[0])
        los = [[0 for _ in range(W)] for __ in range(H)]
        for s in slashers:
            state = s['param1']
            if state != STATE_STUNNED:
                continue
            time_to_wander = s['param0']
            sx, sy = s['x'], s['y']
            los[sy][sx] = max(1 / time_to_wander, los[sy][sx])
        return los

    # ---------- основная логика ----------
    data = {}

    # агент
    agent = None
    for e in entities:
        if e['id'] == player_id:
            agent = e
            break
    assert agent is not None, f"Agent with id {player_id} not found in entities"
    ax, ay = agent['x'], agent['y']
    agent_pos = (ax, ay)

    # базовая карта (окно)
    
    # data['map'] = crop_symbol_map(lines, ax, ay, size)
    data['map'] = crop_map(lines, agent_pos, size, '#', MAP_MAP_NEW)

    # разбор сущностей
    other_explorers, wanderers = [], []
    slashers_xy = []
    explorers = []
    slashers = []
    light_effects = []
    plan_effects = []

    for e in entities:
        k = e.get('kind', '')
        if k == 'EXPLORER':
            if e['id'] != player_id:
                other_explorers.append(e)
            explorers.append(e)
        elif k == 'WANDERER':
            wanderers.append(e)
        elif k == 'EFFECT_LIGHT':
            light_effects.append(e)
        elif k == 'EFFECT_PLAN':
            plan_effects.append(e)
        elif k == 'SLASHER':
            slashers_xy.append((e['x'], e['y']))
            slashers.append(e)

    w_wanderers = [w for w in wanderers if w['param1'] == STATE_WANDERING]
    s_wanderers = [w for w in wanderers if w['param1'] == STATE_SPAWNING]
    del wanderers

    # агрегаты: COUNT
    data['EXPLORER_COUNT']  = norm01_map(count_map(other_explorers, agent_pos, size, clip=3), denom=3.0, invert=False)
    data['WANDERER_COUNT'] = norm01_map(count_map(w_wanderers, agent_pos, size, clip=3), denom=3.0, invert=False)
    data['SLASHER_COUNT'] = norm01_map(count_map(slashers, agent_pos, size, clip=3), denom=3.0, invert=False)

    # агрегаты: MIN-DIST (локальный BFS в окне)
    max_d = 2 * size + 1
    # ally_md  = bfs_min_dist_in_window(lines, other_explorers,  size, agent_pos, max_d=max_d)
    # enemy_md = bfs_min_dist_in_window(lines, w_wanderers, size, agent_pos, max_d=max_d)
    # slashers_md = bfs_min_dist_in_window(lines, slashers_xy, size, agent_pos, max_d=max_d)
    data['EXPLORER_MIN_DIST']  = norm01_map(
        bfs_min_dist_in_window(lines, other_explorers,  size, agent_pos, max_d=max_d),
        denom=max_d, invert=False
    )
    data['WANDERER_MIN_DIST'] = norm01_map(
        bfs_min_dist_in_window(lines, w_wanderers, size, agent_pos, max_d=max_d),
        denom=max_d, invert=False
    )
    # data['SLASHER_MIN_DIST'] = norm01_map(slashers_md, denom=max_d, invert=False)

    data['WANDERER_SPAWNING'] = crop_map(
        build_wanderer_spawning_map(lines, s_wanderers),
        agent_pos, size, pad_val=0,
    )

    # слэшеры: LOS прямо сейчас
    # sl_los_crop = crop_numeric_map(sl_los_full, agent_pos, size, pad_val=0)
    # data['SLASHER_LOS'] = norm01_map(sl_los_crop, denom=1.0, invert=False)  # уже 0/1
    data['SLASHER_STALKING'] = crop_map(
        build_slasher_stalking_los_map(lines, slashers, explorers),
        agent_pos, size, pad_val=0,
    )

    data['SLASHER_WANDERING'] = crop_map(
        build_slasher_wandering_los_map(lines, slashers),
        agent_pos, size, pad_val=0,
    )

    data['SLASHER_SPAWNING'] = crop_map(
        build_slasher_spawning_los_map(lines, slashers),
        agent_pos, size, pad_val=0,
    )

    data['SLASHER_STUNNED'] = crop_map(
        build_slasher_stunned_map(lines, slashers),
        agent_pos, size, pad_val=0,
    )
    
    data['EFFECT_LIGHT'] = norm01_map(
        bfs_min_dist_in_window(lines, light_effects, size, agent_pos, max_d=6),
        denom=6, invert=True
    )
    
    data['EFFECT_PLAN'] = norm01_map(
        bfs_min_dist_in_window(lines, plan_effects, size, agent_pos, max_d=3),
        denom=3, invert=True
    )

    # ваши прежние каналы по FEATURE_ENTITY_DICT (без клиппинга координат!)
    S = 2 * size + 1
    for kind, features in FEATURE_ENTITY_DICT.items():
        for feature in features:
            feat_map = [[0] * S for _ in range(S)]
            border_map = [[0] * S for _ in range(S)]
            if kind == 'EXPLORER':
                feat_map[size][size] = agent.get(feature, 0)
            for e in entities:
                if kind == 'EXPLORER' and e['id'] == agent['id']:
                    continue
                if e.get('kind') != kind:
                    continue
                ex, ey = e['x'], e['y']
                if in_window(ex, ey, ax, ay, size):
                    dx, dy = ex - ax, ey - ay
                    feat_map[dy + size][dx + size] = max(
                        e[feature],
                        feat_map[dy + size][dx + size]
                    )
                else:
                    dx, dy = ex - ax, ey - ay
                    dx = max(min(dx, size), -size)
                    dy = max(min(dy, size), -size)
                    border_map[dy + size][dx + size] = int(e[feature] > 0)
            data[f'{kind}_{feature}'] = feat_map
            if f'{kind}_{feature}' in [
                'EXPLORER_param0',
                'WANDERER_param0',
                'SLASHER_param0',
                'EFFECT_SHELTER_param0',
            ]:
                data[f'{kind}_{feature}_border'] = border_map

    data['EXPLORER_param0'] = norm01_map(data['EXPLORER_param0'], denom=250.0, invert=False)
    data['EXPLORER_param1'] = norm01_map(data['EXPLORER_param1'], denom=3.0, invert=False)
    data['EXPLORER_param2'] = norm01_map(data['EXPLORER_param2'], denom=3.0, invert=False)
    data['WANDERER_param0'] = norm01_map(data['WANDERER_param0'], denom=30.0, invert=False)
    data['EFFECT_SHELTER_param0'] = norm01_map(data['EFFECT_SHELTER_param0'], denom=10.0, invert=False)
    del data['WANDERER_param1']
    del data['SLASHER_param0']
    del data['SLASHER_param1']
    del data['EFFECT_PLAN_param0']
    del data['EFFECT_LIGHT_param0']
    del data['EFFECT_YELL_param0']
    return data


def getActionGreedyMasked(state, Q, action_space_n, mask):
    mask = mask[:action_space_n]
    if not isinstance(state, tuple) or state not in Q:
        ps = np.ma.array(np.ones(action_space_n), mask=mask).filled(0)
        if ps.sum() == 0:
            return np.random.randint(action_space_n)
        return np.random.choice(np.arange(action_space_n), p=ps / ps.sum())
    assert len(Q[state]) == action_space_n
    a = np.ma.array(Q[state], mask=mask)
    a_star = a.argmax()
    return a_star

def calculate_output_np(data, weights, num_classes, softmax=False, num_dirs=5):
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
    if 'out_linear.bias' in weights:
        out_linear_b = weights['out_linear.bias']
    else:
        out_linear_b = 0

    assert data['entity_dir'].shape[-1] == num_dirs
    x_kind_embs = kind_embs[data['entity_kind']]

    entity_features = data['entity_features']
    x_features = entity_features @ features_linear.T + features_linear_b

    entity_dir = data['entity_dir']
    x_dir = entity_dir @ dir_linear.T + dir_linear_b
    entities_mask = (entity_dir > 0).max(axis=-1, keepdims=True)

    x_entitity = np.concatenate((x_kind_embs, x_features, x_dir), axis=-1)
    x = x_entitity @ entity_linear.T + entity_linear_b

    entity_weights = (x_entitity @ entity_impact.T + entity_impact_b) * entities_mask

    # [batch_size, inner_dim, entity_dim]
    x_transposed = x.transpose(0, 2, 1)
    # [batch_size, inner_dim, num_classes]
    x = (x_transposed @ entity_weights)
    # [batch_size, num_classes, inner_dim]
    x = x.transpose(0, 2, 1)
    # [batch_size, num_classes]
    output = (x @ out_linear.T + out_linear_b).squeeze(-1)
    assert output.shape[-1] == num_classes

    if softmax:
        output = sp.softmax(output)
    return output

def get_valid_action_mask_by_coords(e, info):
    # 'UP',
    # 'RIGHT',
    # 'DOWN',
    # 'LEFT',
    # 'WAIT',
    # 'PLAN',
    # 'LIGHT',
    # 'YELL',
    x, y = e['x'], e['y']
    return [
        y > 0 and info['lines'][y - 1][x] != CELL_WALL,
        x < info['width'] - 1 and info['lines'][y][x + 1] != CELL_WALL,
        y < info['height'] - 1 and info['lines'][y + 1][x] != CELL_WALL,
        x > 0 and info['lines'][y][x - 1] != CELL_WALL,
        True,
        e['param1'] > 0 and e['effect_left'] <= 0,
        e['param2'] > 0 and e['effect_left'] <= 0,
        e['can_yell'],
    ]

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


class Solver:
    def __init__(self, info, actions):
        self.info = info
        self.actions = actions
        self.effect_left = 0
        self.yelled = set()
    
    def _check_yell(self, player, entities):
        explorers = [unit for unit in entities[1:] if unit["kind"] == "EXPLORER"]
        for p in explorers:
            p_id = p['id']
            if p_id != player['id'] and \
                p_id not in self.yelled and \
                distance(
                    (player['x'], player['y']), (p['x'], p['y'])
                ) <= 1:
                return True
        return False

    def convert_to_step(self, action_id, player_pos, entities):
        if self.actions[action_id] == 'PLAN':
            self.effect_left = 4
        elif self.actions[action_id] == 'LIGHT':
            self.effect_left = 2
        elif self.actions[action_id] == 'YELL':
            explorers = [unit for unit in entities[1:] if unit["kind"] == "EXPLORER"]
            player = entities[0]
            is_yelled = False
            for p in explorers:
                p_id = p['id']
                if p_id != player['id'] and distance((player['x'], player['y']), (p['x'], p['y'])) <= 1:
                    self.yelled.add(p_id)
                    is_yelled = True
            assert is_yelled

        if action_id >= len(MOVING_KUTULU_ACTIONS):
            return self.actions[action_id]

        rel_pos = MOVE_REL_POS[self.actions[action_id]]
        next_cell = (player_pos[0] + rel_pos[0], player_pos[1] + rel_pos[1])
        return f"MOVE {next_cell[0]} {next_cell[1]}"

    def step(self, entities):
        player_pos = (entities[0]['x'], entities[0]['y'])
        player = entities[0]
        self.effect_left = max(0, self.effect_left - 1)
        player['effect_left'] = self.effect_left
        player['can_yell'] = self._check_yell(player, entities)
        player_mask = ~np.array(get_valid_action_mask_by_coords(player, self.info))
        action_id = self.calculate_action(entities, player_pos, player_mask)
        return self.convert_to_step(action_id, player_pos, entities)

    def calculate_action(self, entities, player_pos, player_mask):
        raise NotImplementedError


class QlearningSolver(Solver):
    def __init__(self, info, actions, Q):
        super(QlearningSolver, self).__init__(info, actions)
        self.Q = Q
        assert list(self.Q.values())[0].shape[0] == len(actions)

    def calculate_action(self, entities, player_pos, player_mask):
        state = get_state(player_pos, entities, self.info['lines'])
        action_id = getActionGreedyMasked(state, self.Q, len(self.actions), player_mask)
        return action_id


class NNSolver(Solver):
    def __init__(self, info, actions, weights: dict):
        super(NNSolver, self).__init__(info, actions)
        self.weights = weights
        self._assert_weights(actions)

    def _assert_weights(self, actions):
        raise NotImplementedError

    def _calculate_output(self, entities, player_pos):
        raise NotImplementedError

    def calculate_action(self, entities, player_pos, player_mask):
        model_output = self._calculate_output(entities, player_pos)
        player_mask = player_mask[:len(self.actions)]
        q_vals_v = np.ma.array(model_output, mask=player_mask)
        action_id = q_vals_v.argmax()
        return action_id


class DQNSolver(NNSolver):
    def _assert_weights(self, actions):
        assert self.weights['entity_impact.weight'].shape[0] == len(actions)

    def _calculate_output(self, entities, player_pos):
        state = get_state_ext(player_pos, entities, self.info['lines'])
        kind_list, features_list, dir_list = parse_state(state)
        data = {
            'entity_kind': [kind_list],
            'entity_features': [features_list],
            'entity_dir': [dir_list],
        }
        model_output = calculate_output_np(data, self.weights, len(self.actions))[0]
        return model_output


class InferenceHelper:
    @staticmethod
    def _relu(x):
        return np.maximum(0, x)

    @staticmethod
    def _maxpool2d(x, kernel_size=2, stride=2):
        N, C, H, W = x.shape
        out_H = H // stride
        out_W = W // stride
        pooled = np.zeros((N, C, out_H, out_W))
        for n in range(N):
            for c in range(C):
                for i in range(out_H):
                    for j in range(out_W):
                        h_start = i * stride
                        w_start = j * stride
                        pooled[n, c, i, j] = np.max(
                            x[n, c, h_start:h_start + kernel_size, w_start:w_start + kernel_size]
                        )
        return pooled

    @staticmethod
    def _conv2d(x, weight, bias, padding=1):
        N, C_in, H, W = x.shape
        C_out, _, kH, kW = weight.shape
        x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
        out = np.zeros((N, C_out, H, W))

        for n in range(N):
            for cout in range(C_out):
                for cin in range(C_in):
                    out[n, cout] += correlate2d(
                        x_padded[n, cin], weight[cout, cin], mode='valid'
                    )
                out[n, cout] += bias[cout]
        return out

    @staticmethod
    def _batchnorm2d(x, mean, var, weight, bias, eps=1e-5):
        # x: [N, C, H, W]
        return weight[None, :, None, None] * ((x - mean[None, :, None, None]) / np.sqrt(var[None, :, None, None] + eps)) + bias[None, :, None, None]

class ConvEncoder:
    def __init__(self):
        self.inference_helper = InferenceHelper()

    def _encode(self, state, weights):
        features = []
        for k in [
            'map',
            'EXPLORER_param0', 'EXPLORER_param1', 'EXPLORER_param2',
            'WANDERER_param0', 'WANDERER_param1',
            'SLASHER_param0', 'SLASHER_param1',
            'EFFECT_PLAN_param0',
            'EFFECT_LIGHT_param0',
            'EFFECT_SHELTER_param0',
            'EFFECT_YELL_param0'
            ]:
            features.append(state[k])
        data = np.array([features])

        conv1 = weights['conv1.weight']
        conv1_b = weights['conv1.bias']
        conv2 = weights['conv2.weight']
        conv2_b = weights['conv2.bias']
        fc1 = weights['fc.weight']
        fc1_b = weights['fc.bias']
        bn1 = weights['bn1.weight']
        bn1_b = weights['bn1.bias']
        bn1_mean = weights['bn1.running_mean']
        bn1_var = weights['bn1.running_var']
        bn2 = weights['bn2.weight']
        bn2_b = weights['bn2.bias']
        bn2_mean = weights['bn2.running_mean']
        bn2_var = weights['bn2.running_var']

        x = data
        x = self.inference_helper._conv2d(x, conv1, conv1_b)
        x = self.inference_helper._batchnorm2d(x, bn1_mean, bn1_var, bn1, bn1_b)
        x = self.inference_helper._relu(x)
        x = self.inference_helper._conv2d(x, conv2, conv2_b)
        x = self.inference_helper._batchnorm2d(x, bn2_mean, bn2_var, bn2, bn2_b)
        x = self.inference_helper._relu(x)
        x = self.inference_helper._maxpool2d(x)

        # # Flatten
        N = x.shape[0]
        x = x.reshape(N, -1)

        # # FC layers
        x = self.inference_helper._relu(np.dot(x, fc1.T) + fc1_b)
        
        return x

class ConvExtEncoder:
    def __init__(self):
        self.inference_helper = InferenceHelper()

    def _encode(self, state, weights):
        features = []
        for k in [
            'map',
            'EXPLORER_param0', 'EXPLORER_param1', 'EXPLORER_param2',
            'WANDERER_param0', 'WANDERER_param1',
            'SLASHER_param0', 'SLASHER_param1',
            'EFFECT_PLAN_param0',
            'EFFECT_LIGHT_param0',
            'EFFECT_SHELTER_param0',
            'EFFECT_YELL_param0',
            'EXPLORER_COUNT',
            'EXPLORER_MIN_DIST',
            'WANDERER_COUNT',
            'WANDERER_MIN_DIST',
            'SLASHER_COUNT',
            'SLASHER_STALKING',
            'SLASHER_WANDERING',
            'SLASHER_SPAWNING',
            'SLASHER_STUNNED',
            ]:
            features.append(state[k])
        data = np.array([features])

        conv1 = weights['conv1.weight']
        conv1_b = weights['conv1.bias']
        conv2 = weights['conv2.weight']
        conv2_b = weights['conv2.bias']
        fc1 = weights['fc.weight']
        fc1_b = weights['fc.bias']
        bn1 = weights['bn1.weight']
        bn1_b = weights['bn1.bias']
        bn1_mean = weights['bn1.running_mean']
        bn1_var = weights['bn1.running_var']
        bn2 = weights['bn2.weight']
        bn2_b = weights['bn2.bias']
        bn2_mean = weights['bn2.running_mean']
        bn2_var = weights['bn2.running_var']

        x = data
        x = self.inference_helper._conv2d(x, conv1, conv1_b)
        x = self.inference_helper._batchnorm2d(x, bn1_mean, bn1_var, bn1, bn1_b)
        x = self.inference_helper._relu(x)
        x = self.inference_helper._conv2d(x, conv2, conv2_b)
        x = self.inference_helper._batchnorm2d(x, bn2_mean, bn2_var, bn2, bn2_b)
        x = self.inference_helper._relu(x)
        x = self.inference_helper._maxpool2d(x)

        # # Flatten
        N = x.shape[0]
        x = x.reshape(N, -1)

        # # FC layers
        x = self.inference_helper._relu(np.dot(x, fc1.T) + fc1_b)
        
        return x


class DQNConvSolver(NNSolver):
    def __init__(self, info, actions, weights, size=3):
        super(DQNConvSolver, self).__init__(info, actions, weights)
        self.size = size
        self.conv_encoder = ConvEncoder()

    def _assert_weights(self, actions):
        assert self.weights['fc2.weight'].shape[0] == len(actions)

    def _calculate_output(self, entities, player_pos):
        # player_pos is already provided as a parameter, so we don't need to extract it from entities
        player_id = entities[0]['id']
        state = get_state_conv(player_id, entities, self.info['lines'], self.size)

        fc2 = self.weights['fc2.weight']
        fc2_b = self.weights['fc2.bias']

        x = self.conv_encoder._encode(state, self.weights)
        x = self.conv_encoder.inference_helper._relu(x)
        x = np.dot(x, fc2.T) + fc2_b

        return x[0]


class DQNByKindSolver(NNSolver):
    def __init__(self, info, actions, weights):
        super(DQNByKindSolver, self).__init__(info, actions, weights)
        self.entity_kinds = set([key.split('.')[1] for key in weights.keys()])
        self.weights_dict = {
            kind: {
                k[len(f"model_by_kind.{kind}."):]: v
                for k, v in weights.items()
                if k.startswith(f"model_by_kind.{kind}.")
            }
            for kind in self.entity_kinds
        }

    def _assert_weights(self, actions):
        assert self.weights['model_by_kind.WANDERER.entity_impact.weight'].shape[0] == len(actions)
        for key, val in self.weights.items():
            if key.endswith('entity_impact.weight'):
                assert val.shape[0] == len(actions), f'wrong shape for {key} layers'

    def _calculate_output(self, entities, player_pos):
        state = get_state_ext(player_pos, entities, self.info['lines'])
        data_by_kind = {}
        state_by_kind = parse_state_by_kind(state)
        for kind, (kind_list, features_list, dir_list) in state_by_kind.items():
            if kind not in data_by_kind:
                data_by_kind[kind] = {
                    'entity_kind': [],
                    'entity_features': [],
                    'entity_dir': [],
                }
            data_by_kind[kind]['entity_kind'].append(kind_list)
            data_by_kind[kind]['entity_features'].append(features_list)
            data_by_kind[kind]['entity_dir'].append(dir_list)
        
        model_outputs = []
        for kind in self.entity_kinds:
            weights = self.weights_dict[kind]
            data = data_by_kind[kind]
            model_output = calculate_output_np(data, weights, len(self.actions))[0]
            model_outputs.append(model_output)
        
        all_model_output = np.array(model_outputs).sum(axis=0)
        return all_model_output


class PPOConvSolver(NNSolver):
    def __init__(self, info, actions, weights, size=3):
        super(PPOConvSolver, self).__init__(info, actions, weights)
        self.size = size
        self.conv_encoder = ConvEncoder()

    def _assert_weights(self, actions):
        assert self.weights['actor.weight'].shape[0] == len(actions)

    def _calculate_output(self, entities, player_pos):
        # player_pos is already provided as a parameter, so we don't need to extract it from entities
        player_id = entities[0]['id']
        state = get_state_conv(player_id, entities, self.info['lines'], self.size)

        a = self.weights['actor.weight']
        a_b = self.weights['actor.bias']

        x = self.conv_encoder._encode(state, self.weights)
        x = self.conv_encoder.inference_helper._relu(x)
        x = np.dot(x, a.T) + a_b
        x = sp.softmax(x)
        return x[0]


class PPOConvExtSolver(NNSolver):
    def __init__(self, info, actions, weights, size=3):
        super(PPOConvExtSolver, self).__init__(info, actions, weights)
        self.size = size
        self.conv_encoder = ConvExtEncoder()

    def _assert_weights(self, actions):
        assert self.weights['actor.weight'].shape[0] == len(actions)

    def _calculate_output(self, entities, player_pos):
        # player_pos is already provided as a parameter, so we don't need to extract it from entities
        player_id = entities[0]['id']
        state = get_state_conv_ext(player_id, entities, self.info['lines'], self.size)

        a = self.weights['actor.weight']
        a_b = self.weights['actor.bias']

        x = self.conv_encoder._encode(state, self.weights)
        x = self.conv_encoder.inference_helper._relu(x)
        x = np.dot(x, a.T) + a_b
        x = sp.softmax(x)
        return x[0]


def main():
    vals = pkl.loads(zlib.decompress(base64.b64decode(data1)))
    vals = [np.load(io.BytesIO(byte_data)) for byte_data in vals]
    keys = pkl.loads(zlib.decompress(base64.b64decode(data2)))
    checkpoint_data = dict(zip(keys, vals))
    info = parse_info()
    if mode == 'qlearning':
        solver = QlearningSolver(info, USED_ACTIONS, checkpoint_data)
    elif mode == 'dqn_ext':
        solver = DQNSolver(info, USED_ACTIONS, checkpoint_data)
    elif mode == 'dqn_by_kind':
        solver = DQNByKindSolver(info, USED_ACTIONS, checkpoint_data)
    elif mode == 'dqn_conv':
        solver = DQNConvSolver(info, USED_ACTIONS, checkpoint_data, SIZE)
    elif mode == 'ppo_conv':
        solver = PPOConvSolver(info, USED_ACTIONS, checkpoint_data, SIZE)
    elif mode == 'ppo_conv_ext':
        solver = PPOConvExtSolver(info, USED_ACTIONS, checkpoint_data, SIZE)
    else:
        raise ValueError(f'unknown mode: "{mode}"')
    
    # game loop
    while True:
        entity_count = int(input())  # the first given entity corresponds to your explorer
        entities = []
        for i in range(entity_count):
            entities.append(parse_desc(input()))

        step = solver.step(entities)
        print(step)

if __name__ == '__main__':
    main()
