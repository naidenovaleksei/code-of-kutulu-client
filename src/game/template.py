import sys
import io
import math
import numpy as np
import scipy.special as sp
from scipy.signal import correlate2d
import heapq
from collections import defaultdict

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
