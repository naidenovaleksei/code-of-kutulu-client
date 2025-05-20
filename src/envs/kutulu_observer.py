from src.envs.kutulu_world import KutuluWorldEnv
from src.game.template import (
    get_state,
    get_distances,
    get_state_ext,
)


class BaseKutuluClosestObserver:
    def __init__(self, env: KutuluWorldEnv):
        self.env = env
    
    def get_state(self) -> tuple:
        raise NotImplementedError


class KutuluClosestObserver(BaseKutuluClosestObserver):
    def __init__(self, env: KutuluWorldEnv):
        super(KutuluClosestObserver, self).__init__(env)

    def get_state(self, player_id, state_type):
        _obs = self.env._get_obs(player_id)
        entities = _obs['entities']
        player_pos = (entities[0]['x'], entities[0]['y'])
        
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

    def get_state(self, player_id, state_type):
        _obs = self.env._get_obs(player_id)
        entities = _obs['entities']
        player_pos = (entities[0]['x'], entities[0]['y'])
        
        state = get_state_ext(player_pos, entities, self.env.map,
                          get_distances_func=self.get_distances_cached())
        return state
