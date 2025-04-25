from src.envs.kutulu_world import KutuluWorldEnv

move_to_rel_pos = {
    'UP': (0,-1),
    'RIGHT': (1,0),
    'DOWN': (0,1),
    'LEFT': (-1,0),
    'WAIT': (0,0)
}
rel_pos_to_move = {v: k for k,v in move_to_rel_pos.items()}
action_idx = list(move_to_rel_pos.values())


class KutuluPlayer:
    def __init__(self, env: KutuluWorldEnv):
        self.env = env

    def get_observations(self):
        observations = {}
        for player in self.env.active_players():
            player_id = int(player['id'])
            _obs = self.env._get_obs(player_id)
            entities = _obs['entities']
            player_pos = (entities[0]['x'], entities[0]['y'])
            explorers = [unit for unit in entities[1:] if unit["type"] == "EXPLORER"]
            wanderers = [unit for unit in entities[1:] if unit["type"] == "WANDERER" and unit["wandering"] == 1]
            explorer_distances = self._get_distances(explorers, player_pos)
            wanderers_distances = self._get_distances(wanderers, player_pos)
            closest_explorer_dir, closest_explorer_dist = self._get_min_direction_and_distance(explorer_distances)
            closest_wanderer_dir, closest_wanderer_dist = self._get_min_direction_and_distance(wanderers_distances)
            observations[player_id] = {
                'closest_explorer_dir': closest_explorer_dir,
                'closest_explorer_dist': closest_explorer_dist,
                'closest_wanderer_dir': closest_wanderer_dir,
                'closest_wanderer_dist': closest_wanderer_dist,
            }
        return observations

    def _get_distances(self, entities, player_pos):
        distances = {}
        for _, rel_pos in move_to_rel_pos.items():
            pos = (rel_pos[0] + player_pos[0], rel_pos[1] + player_pos[1])
            if not 0 < pos[0] < len(self.env.map[0]) - 1:
                continue
            if not 0 < pos[1] < len(self.env.map) - 1:
                continue
            for e in entities:
                entity_pos = (e['x'], e['y'])
                path = self.env._find_path_cached(pos, entity_pos)
                if len(path):
                    assert path[-1] != pos
                    assert path[0] == entity_pos
                distances[rel_pos] = min(distances.get(rel_pos, 1000), len(path))
        return distances

    def _get_min_direction_and_distance(self, distances):
        if len(distances) == 0:
            return None, None
        d_min = min(distances.values())
        return '_'.join([rel_pos_to_move[k] for k, v in distances.items() if v == d_min]), d_min