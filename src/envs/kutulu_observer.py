from src.envs.kutulu_world import KutuluWorldEnv
from src.game.template import (
    get_state,
    get_distances,
    get_state_ext,
    get_state_ext_v2,
    REL_POSITIONS,
    get_state_conv,
    get_state_conv_ext,
)


# public static final char WALL 			= '#';
# public static final char EMPTY 			= '.';
# public static final char WANDERER_SPAWN 	= 'w';
# public static final char OTHER_SPAWN 	= 's';
# public static final char PLAYER_SPAWN 	= 'S';
# public static final char SHELTER 		= 'U';
VIZ_MAP = {
    "EXPLORER": 'E',
    "WANDERER": 'W',
    "SLASHER": 'R',
    "EFFECT_PLAN": 'P',
    "EFFECT_LIGHT": 'L',
    "EFFECT_SHELTER": 'H',
    "EFFECT_YELL": 'Y',
}


def viz_observation(observation, action=None):
    lines = observation['info']['lines']
    entities = observation['obs']['entities']
    player_id = observation['player_id']

    curr_map = [list(line) for line in lines]
    for e in entities:
        if e['id'] == player_id:
            agent_pos = (e['x'], e['y'])
            curr_map[e['y']][e['x']] = 'A'
        elif e['kind'] == 'EXPLORER':
            curr_map[e['y']][e['x']] = str(e['id'])
        else:
            curr_map[e['y']][e['x']] = VIZ_MAP[e['kind']]
    if action is not None and action < len(REL_POSITIONS):
        rel_pos = REL_POSITIONS[action]
        x = agent_pos[0] + rel_pos[0]
        y = agent_pos[1] + rel_pos[1]
        curr_map[y][x] = '^'
    for line in curr_map:
        print(''.join(line))
    print()


def viz_observation_cls(observation, action=None):
    lines = observation.info.lines
    entities = observation.obs.entities
    player_id = observation.player_id

    curr_map = [list(line) for line in lines]
    for e in entities:
        if e.id == player_id:
            agent_pos = (e.x, e.y)
            curr_map[e.y][e.x] = 'A'
        elif e.kind == 'EXPLORER':
            curr_map[e.y][e.x] = str(e.id)
        else:
            curr_map[e.y][e.x] = VIZ_MAP[e.kind]
    if action is not None and action < len(REL_POSITIONS):
        rel_pos = REL_POSITIONS[action]
        x = agent_pos[0] + rel_pos[0]
        y = agent_pos[1] + rel_pos[1]
        curr_map[y][x] = '^'
    for line in curr_map:
        print(''.join(line))
    print()


class BaseKutuluClosestObserver:
    def __init__(self, env: KutuluWorldEnv):
        self.env = env
    
    def get_state(self, player_id) -> tuple:
        raise NotImplementedError


class KutuluRawObserver(BaseKutuluClosestObserver):
    def __init__(self, env: KutuluWorldEnv):
        super(KutuluRawObserver, self).__init__(env)

    def get_state(self, player_id: int):
        entities = self.env.get_obs(player_id).entities
        assert entities[0].id == player_id
        entities = [e.to_dict() for e in entities]
        state = entities
        return state


class KutuluClosestObserver(BaseKutuluClosestObserver):
    def __init__(self, env: KutuluWorldEnv):
        super(KutuluClosestObserver, self).__init__(env)

    def get_state(self, player_id: int):
        entities = self.env.get_obs(player_id).entities
        assert entities[0].id == player_id
        player_pos = (entities[0].x, entities[0].x)
        entities = [e.to_dict() for e in entities]
        state = get_state(player_pos, entities, self.env.map,
                          get_distances_func=self.get_distances_cached())
        return state

    def find_path_cached(self):
        def find_path_cached_(pos, entity_pos, lines):
            return self.env.find_path_cached(pos, entity_pos)
        return find_path_cached_

    def _get_distances(self, entities, player_pos):
        distances = get_distances(entities, player_pos, self.env.map,
                                  find_path_func=self.find_path_cached())
        return distances

    def get_distances_cached(self):
        def get_distances_cached_(entities, player_pos, lines):
            return self._get_distances(entities, player_pos)
        return get_distances_cached_


class KutuluClosestExtObserver(KutuluClosestObserver):
    def __init__(self, env: KutuluWorldEnv):
        super(KutuluClosestExtObserver, self).__init__(env)

    def get_state(self, player_id: int):
        entities = self.env.get_obs(player_id).entities
        assert entities[0].id == player_id
        player_pos = (entities[0].x, entities[0].x)
        entities = [e.to_dict() for e in entities]
        state = get_state_ext(player_pos, entities, self.env.map,
                          get_distances_func=self.get_distances_cached())
        return state


class KutuluClosestExtv2Observer(KutuluClosestObserver):
    def __init__(self, env: KutuluWorldEnv):
        super(KutuluClosestExtv2Observer, self).__init__(env)

    def get_state(self, player_id: int):
        entities = self.env.get_obs(player_id).entities
        assert entities[0].id == player_id
        entities = [e.to_dict() for e in entities]
        state = get_state_ext_v2(player_id, entities, self.env.map,
                          get_distances_func=self.get_distances_cached())
        return state


class KutuluConvObserver(BaseKutuluClosestObserver):
    def __init__(self, env: KutuluWorldEnv, size: int):
        super(KutuluConvObserver, self).__init__(env)
        self.size = size

    def get_state(self, player_id: int):
        entities = self.env.get_obs(player_id).entities
        assert entities[0].id == player_id
        entities = [e.to_dict() for e in entities]
        state = get_state_conv(player_id, entities, self.env.map, self.size)
        return state


class KutuluConvExtObserver(BaseKutuluClosestObserver):
    def __init__(self, env: KutuluWorldEnv, size: int):
        super(KutuluConvExtObserver, self).__init__(env)
        self.size = size

    def get_state(self, player_id: int):
        entities = self.env.get_obs(player_id).entities
        assert entities[0].id == player_id
        entities = [e.to_dict() for e in entities]
        state = get_state_conv_ext(player_id, entities, self.env.map, self.size)
        return state
