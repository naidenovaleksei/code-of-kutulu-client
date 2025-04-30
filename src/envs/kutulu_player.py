from src.envs.kutulu_world import KutuluWorldEnv, CELL_WALL

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


class BaseKutuluPlayer:
    def __init__(self, env: KutuluWorldEnv):
        self.env = env
    
    def get_state(self) -> tuple:
        raise NotImplementedError

class KutuluPlayer(BaseKutuluPlayer):
    def __init__(self, env: KutuluWorldEnv):
        super(KutuluPlayer, self).__init__(env)

    def get_state(self, player_id, max_explorer_dist=999, max_wanderer_dist=999):
        _obs = self.env._get_obs(player_id)
        entities = _obs['entities']
        player_pos = (entities[0]['x'], entities[0]['y'])
        explorers = [unit for unit in entities[1:] if unit["type"] == "EXPLORER"]
        wanderers = [unit for unit in entities[1:] if unit["type"] == "WANDERER" and unit["wandering"] == 1]
        explorer_distances = self._get_distances(explorers, player_pos)
        wanderers_distances = self._get_distances(wanderers, player_pos)
        closest_explorer_dir, closest_explorer_dist = self._get_min_direction_and_distance(explorer_distances)
        closest_wanderer_dir, closest_wanderer_dist = self._get_min_direction_and_distance(wanderers_distances)
        if closest_explorer_dist is not None:
            closest_explorer_dist = min(closest_explorer_dist, max_explorer_dist)
        if closest_wanderer_dist is not None:
            closest_wanderer_dist = min(closest_wanderer_dist, max_wanderer_dist)
        return (
            closest_explorer_dir,
            closest_explorer_dist,
            closest_wanderer_dir,
            closest_wanderer_dist,
        )

    def get_observations(self):
        observations = {}
        for player in self.env.active_players():
            player_id = int(player['id'])
            edir, edist, wdir, wdist = self.get_state(int(player['id']))
            observations[player_id] = {
                'closest_explorer_dir': edir,
                'closest_explorer_dist': edist,
                'closest_wanderer_dir': wdir,
                'closest_wanderer_dist': wdist,
            }

        return observations

    def _get_distances(self, entities, player_pos):
        distances = {}
        for rel_pos in REL_POSITIONS:
            pos = (rel_pos[0] + player_pos[0], rel_pos[1] + player_pos[1])
            if not 0 < pos[0] < len(self.env.map[0]) - 1:
                continue
            if not 0 < pos[1] < len(self.env.map) - 1:
                continue
            if self.env.map[pos[1]][pos[0]] == CELL_WALL:
                continue
            for e in entities:
                entity_pos = (e['x'], e['y'])
                if entity_pos == player_pos:
                    return {MOVE_REL_POS['WAIT']: 0}
                path = self.env.find_path_cached(pos, entity_pos)
                if len(path):
                    assert path[-1] != pos
                    assert path[0] == entity_pos
                distances[rel_pos] = min(distances.get(rel_pos, 1000), len(path))
        return distances

    def _get_min_direction_and_distance(self, distances):
        if len(distances) == 0:
            return None, None
        # distances = {rel_pos: d for rel_pos, d in distances.items()}
        d_min = min(distances.values())
        key = tuple(
            i
            for i, rel_pos in enumerate(REL_POSITIONS)
            if distances.get(rel_pos) == d_min
        )
        # if len(key) == 4:
        #     # (0, 1, 2, 3,)
        #     return (4,), 0
        return key, d_min + REL_SHIFT[REL_POSITIONS[key[0]]]
