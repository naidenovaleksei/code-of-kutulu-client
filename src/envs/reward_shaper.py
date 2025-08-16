from typing import List, Tuple

import numpy as np

from src.envs.kutulu_entities import (
    EntityKind,
    EffectType,
    KutuluEntity,
    MoveType,
)
from src.game.template import (
    REL_POSITIONS,
    get_all_distances,
)
from src.envs.agents import AgentObservation
from src.envs.distance import find_path, distance, UnreachedPositionError

PLAN_DISTANCE = 2
LIGHT_DISTANCE = 5
YELL_DISTANCE = 1

class PotentialRewardShaper:
    def __init__(self, actions, verbose,
                 gamma=0.99,
                 lights_left_coef=0.04,
                 light_potential_coef=1.0,
                 plans_left_coef=0.05,
                 plan_potential_coef=1.0,
                 yell_potential_coef=1.0,
                 w_nearby_potential_coef=1.0,
                 e_nearby_potential_coef=1.0,
                 s_nearby_potential_coef=1.0,
                 sanity_loss_potential_coef=0.1,
                 no_move_reward_coef=0.5,
            ):
        self.actions = actions
        self.verbose = verbose
        self.lights_left_coef = lights_left_coef
        self.light_gamma = gamma
        self.light_potential_coef = light_potential_coef
        self.plans_left_coef = plans_left_coef
        self.plan_gamma = gamma
        self.plan_potential_coef = plan_potential_coef
        self.yell_gamma = gamma
        self.yell_potential_coef = yell_potential_coef
        self.w_nearby_gamma = gamma
        self.w_nearby_potential_coef = w_nearby_potential_coef
        self.e_nearby_gamma = gamma
        self.e_nearby_potential_coef = e_nearby_potential_coef
        self.s_nearby_gamma = gamma
        self.s_nearby_potential_coef = s_nearby_potential_coef
        self.sanity_loss_gamma = gamma
        self.sanity_loss_potential_coef = sanity_loss_potential_coef
        self.no_move_reward_coef = no_move_reward_coef

    def _get_wanderers(self, observation: AgentObservation, player_id=None):
        return [
            e for e in observation.obs.entities
            if e.kind in (EntityKind.WANDERER.value, EntityKind.SLASHER.value) and \
                (player_id is None or e.param2 == player_id)
        ]
    
    def _get_explorers(self, observation, player_id) -> List[KutuluEntity]:
        return [
            e for e in observation.obs.entities
            if e.kind == EntityKind.EXPLORER.value and e.id != player_id
        ]
    
    def _get_shelters(self, observation) -> List[KutuluEntity]:
        return [
            e for e in observation.obs.entities
            if e.kind == EntityKind.EFFECT_SHELTER.value and e.param0 > 0
        ]

    def _light_potential(self, observation: AgentObservation):
        # more - better
        player = observation.obs.entities[0]
        assert player.id == observation.player_id
        lights_left = player.param2
        wanderers_nearby_potential = self._wanderers_nearby_potential(observation, LIGHT_DISTANCE) 
        return wanderers_nearby_potential + self.lights_left_coef * lights_left

    def _plan_potential(self, observation: AgentObservation):
        # more - better
        player = observation.obs.entities[0]
        assert player.id == observation.player_id
        player_pos = (player.x, player.y)
        explorers = self._get_explorers(observation, player.id)
        sum_sanity = player.param0
        for e in explorers:
            e_pos = (e.x, e.y)
            if distance(player_pos, e_pos) <= PLAN_DISTANCE:
                if len(find_path(player_pos, e_pos, observation.info.lines)) <= PLAN_DISTANCE:
                    sum_sanity += e.param0
        plans_left = player.param1
        # distance[1..1000], plans_left[0..3] -> potential[0.015..0.4, 0] + 0.05 * [0..3]
        # TODO: sqrt or log?
        return np.sqrt(sum_sanity) + self.plans_left_coef * plans_left

    def _wanderers_nearby_potential(self, observation: AgentObservation, limit=4):
        # more - better
        player = observation.obs.entities[0]
        assert player.id == observation.player_id
        player_pos = (player.x, player.y)
        wanderers = self._get_wanderers(observation)
        wanderers_distances = []
        for w in wanderers:
            w_pos = (w.x, w.y)
            if distance(player_pos, w_pos) <= limit:
                w_distance = len(find_path(player_pos, w_pos, observation.info.lines))
                if w_distance <= limit:
                    wanderers_distances.append(w_distance)
        # TODO: 1 / (1 + x) or exp(-x)?
        # return - (1 / (1 + np.array(wanderers_distances))).sum()
        return - np.exp(-np.array(wanderers_distances)).sum()

    def _explorers_nearby_potential(self, observation: AgentObservation):
        # more - better
        player = observation.obs.entities[0]
        assert player.id == observation.player_id
        player_pos = (player.x, player.y)
        explorers = self._get_explorers(observation, player.id)
        explorers_distances = []
        for w in explorers:
            w_pos = (w.x, w.y)
            w_distance = len(find_path(player_pos, w_pos, observation.info.lines))
            explorers_distances.append(w_distance)
        return (1 / (1 + np.array(explorers_distances))).sum()
        # return np.exp(-np.array(explorers_distances)).sum()

    def _shelters_nearby_potential(self, observation: AgentObservation):
        # more - better
        player = observation.obs.entities[0]
        assert player.id == observation.player_id
        player_pos = (player.x, player.y)
        shelters = self._get_shelters(observation)
        shelters_distances = []
        for s in shelters:
            s_pos = (s.x, s.y)
            try:
                s_distance = len(find_path(player_pos, s_pos, observation.info.lines))
                shelters_distances.append(s_distance)
            except UnreachedPositionError:
                pass
        if len(shelters_distances) == 0:
            return 0
        return 1 / (1 + min(shelters_distances))
        # return np.exp(-min(shelters_distances))

    def _others_sanity_loss_potential(self, observation: AgentObservation, distance_limit: int):
        # more - better
        player = observation.obs.entities[0]
        assert player.id == observation.player_id
        player_pos = (player.x, player.y)
        explorers = self._get_explorers(observation, player.id)
        sum_sanity = 1
        for e in explorers:
            e_pos = (e.x, e.y)
            if distance(player_pos, e_pos) <= distance_limit:
                sum_sanity += e.param0
        # distance[0..750] -> potential[-0.015..-0.4, 0]
        # TODO: sqrt or log?
        return -np.log(sum_sanity)

    def _light_bonus(self, observation, next_observation):
        light_potential = self._light_potential(observation)
        next_light_potential = self._light_potential(next_observation)
        light_bonus = self.light_gamma * next_light_potential - light_potential
        return light_bonus * self.light_potential_coef

    def _plan_bonus(self, observation, next_observation):
        plan_potential = self._plan_potential(observation)
        next_plan_potential = self._plan_potential(next_observation)
        plan_bonus = self.plan_gamma * next_plan_potential - plan_potential
        return plan_bonus * self.plan_potential_coef

    def _yell_bonus(self, observation, next_observation):
        yell_potential = self._others_sanity_loss_potential(observation, YELL_DISTANCE)
        next_yell_potential = self._others_sanity_loss_potential(next_observation, YELL_DISTANCE)
        yell_bonus = self.yell_gamma * next_yell_potential - yell_potential
        yell_bonus = max(yell_bonus, 0)
        return yell_bonus * self.yell_potential_coef

    def _wanderers_nearby_bonus(self, observation, next_observation):
        w_nearby_potential = self._wanderers_nearby_potential(observation)
        next_w_nearby_potential = self._wanderers_nearby_potential(next_observation)
        w_nearby_bonus = self.w_nearby_gamma * next_w_nearby_potential - w_nearby_potential
        return w_nearby_bonus * self.w_nearby_potential_coef

    def _explorers_nearby_bonus(self, observation, next_observation):
        e_nearby_potential = self._explorers_nearby_potential(observation)
        next_e_nearby_potential = self._explorers_nearby_potential(next_observation)
        e_nearby_bonus = self.e_nearby_gamma * next_e_nearby_potential - e_nearby_potential
        return e_nearby_bonus * self.e_nearby_potential_coef

    def _shelters_nearby_bonus(self, observation, next_observation):
        s_nearby_potential = self._shelters_nearby_potential(observation)
        next_s_nearby_potential = self._shelters_nearby_potential(next_observation)
        s_nearby_bonus = self.s_nearby_gamma * next_s_nearby_potential - s_nearby_potential
        return s_nearby_bonus * self.s_nearby_potential_coef

    def _others_sanity_loss_bonus(self, observation, next_observation):
        sanity_loss_potential = self._others_sanity_loss_potential(observation, 3)
        next_sanity_loss_potential = self._others_sanity_loss_potential(next_observation, 3)
        sanity_loss_bonus = self.sanity_loss_gamma * next_sanity_loss_potential - sanity_loss_potential
        sanity_loss_bonus = max(sanity_loss_bonus, 0)
        return sanity_loss_bonus * self.sanity_loss_potential_coef

    def _compute_shaped_reward(self,
                               observation: AgentObservation,
                               action: int,
                               next_observations: AgentObservation,
                               original_reward: float,
                               other_rewards: List[float]):
        reward = 0
        bonuses = {
            'original_reward': original_reward,
            'no_move_bonus': 0,
            'wanderers_nearby_bonus': 0,
            'explorers_nearby_bonus': 0,
            'shelters_nearby_bonus': 0,
            'others_sanity_loss_bonus': 0,
            'plan_bonus': 0,
            'light_bonus': 0,
            'yell_bonus': 0
        }
        
        if original_reward < 0 and self.actions[action] not in (
            MoveType.UP.value,
            MoveType.RIGHT.value,
            MoveType.DOWN.value,
            MoveType.LEFT.value,
        ):
            no_move_bonus = self.no_move_reward_coef * original_reward
            reward += no_move_bonus
            bonuses['no_move_bonus'] = no_move_bonus
            if self.verbose and abs(no_move_bonus) > 1e-4:
                print(f"no_move_bonus: {no_move_bonus}")
            return reward, bonuses

        wanderers_nearby_bonus = self._wanderers_nearby_bonus(observation, next_observations)
        reward += wanderers_nearby_bonus
        bonuses['wanderers_nearby_bonus'] = wanderers_nearby_bonus
        if self.verbose and abs(wanderers_nearby_bonus) > 1e-4:
            print(f"wanderers_nearby_bonus: {wanderers_nearby_bonus}")

        explorers_nearby_bonus = self._explorers_nearby_bonus(observation, next_observations)
        reward += explorers_nearby_bonus
        bonuses['explorers_nearby_bonus'] = explorers_nearby_bonus
        if self.verbose and abs(explorers_nearby_bonus) > 1e-4:
            print(f"explorers_nearby_bonus: {explorers_nearby_bonus}")

        shelters_nearby_bonus = self._shelters_nearby_bonus(observation, next_observations)
        reward += shelters_nearby_bonus
        bonuses['shelters_nearby_bonus'] = shelters_nearby_bonus
        if self.verbose and abs(shelters_nearby_bonus) > 1e-4:
            print(f"shelters_nearby_bonus: {shelters_nearby_bonus}")

        others_sanity_loss_bonus = self._others_sanity_loss_bonus(observation, next_observations)
        reward += others_sanity_loss_bonus
        bonuses['others_sanity_loss_bonus'] = others_sanity_loss_bonus
        if self.verbose and abs(others_sanity_loss_bonus) > 1e-4:
            print(f"others_sanity_loss_bonus: {others_sanity_loss_bonus}")

        if self.actions[action] == EffectType.PLAN.value:
            assert original_reward >= 0
            plan_bonus = self._plan_bonus(observation, next_observations)
            reward += plan_bonus
            bonuses['plan_bonus'] = plan_bonus
            if self.verbose and abs(plan_bonus) > 1e-4:
                print(f"plan_bonus: {plan_bonus}")
        elif self.actions[action] == EffectType.LIGHT.value:
            assert original_reward >= 0
            light_bonus = self._light_bonus(observation, next_observations)
            reward += light_bonus
            bonuses['light_bonus'] = light_bonus
            if self.verbose and abs(light_bonus) > 1e-4:
                print(f"light_bonus: {light_bonus}")
        elif self.actions[action] == EffectType.YELL.value:
            assert original_reward >= 0
            yell_bonus = self._yell_bonus(observation, next_observations)
            reward += yell_bonus
            bonuses['yell_bonus'] = yell_bonus
            if self.verbose and abs(yell_bonus) > 1e-4:
                print(f"yell_bonus: {yell_bonus}")
        return reward, bonuses

    def recalculate_rewards(self,
                            rewards: List[float],
                            actions: List[int],
                            states: List,
                            dones: List[bool],
                            other_rewards: List[List[float]],
                            observations: List):
        assert dones[-1]
        assert sum(dones[:-1]) == 0
        rewards = list(rewards)
        T = len(rewards)
        
        # Initialize bonus tracking
        bonus_sums = {
            'original_reward': 0,
            'no_move_bonus': 0,
            'wanderers_nearby_bonus': 0,
            'explorers_nearby_bonus': 0,
            'shelters_nearby_bonus': 0,
            'others_sanity_loss_bonus': 0,
            'plan_bonus': 0,
            'light_bonus': 0,
            'yell_bonus': 0
        }
        bonus_counts = {key: 0 for key in bonus_sums}
        
        for t in range(0, T - 1):
            shaped_reward, bonuses = self._compute_shaped_reward(
                observations[t], actions[t], observations[t+1], rewards[t], other_rewards[t],
            )
            rewards[t] += shaped_reward
            
            # Accumulate bonuses
            for bonus_type, value in bonuses.items():
                if value != 0:  # Only count non-zero bonuses
                    bonus_sums[bonus_type] += value
                    bonus_counts[bonus_type] += 1
        
        # Calculate average bonuses
        avg_bonuses = {}
        for bonus_type, total in bonus_sums.items():
            count = bonus_counts[bonus_type]
            avg_bonuses[bonus_type] = total / max(1, count)  # Avoid division by zero
            
        return rewards, avg_bonuses



class RewardShaper:
    def __init__(self, actions, verbose,
            good_plan_bonus=0.3,
            bad_plan_bonus=-0.2,
            good_light_bonus=0.3,
            bad_light_bonus=-0.1,
            other_reward_coef=0.01,
            good_explorers_nearby_bonus=0.02,
            bad_explorers_nearby_bonus=-0.04,
            yell_bonus_coef=0.5,
            bad_yell_bonus=-0.2,
            shelter_bonus=0.5,
            wait_reward_coef=1.0,
            bad_towards_enemy_bonus=-0.3,
            ):
        self.actions = actions
        self.verbose = verbose
        self.good_plan_bonus = good_plan_bonus
        self.bad_plan_bonus = bad_plan_bonus
        self.good_light_bonus = good_light_bonus
        self.bad_light_bonus = bad_light_bonus
        self.other_reward_coef = other_reward_coef
        self.good_explorers_nearby_bonus = good_explorers_nearby_bonus
        self.bad_explorers_nearby_bonus = bad_explorers_nearby_bonus
        self.bad_yell_bonus = bad_yell_bonus
        self.yell_bonus_coef = yell_bonus_coef
        self.shelter_bonus = shelter_bonus
        self.wait_reward_coef = wait_reward_coef
        self.bad_towards_enemy_bonus = bad_towards_enemy_bonus

    def _get_wanderers(self, observation):
        return [
            e for e in observation.obs.entities
            if e.kind in (EntityKind.WANDERER.value, EntityKind.SLASHER.value)
        ]
    
    def _get_explorers(self, observation, player_id) -> List[KutuluEntity]:
        return [
            e for e in observation.obs.entities
            if e.kind == EntityKind.EXPLORER.value and e.id != player_id
        ]
    
    def _get_shelters(self, observation) -> List[KutuluEntity]:
        return [
            e for e in observation.obs.entities
            if e.kind == EntityKind.EFFECT_SHELTER.value
        ]
    
    def _get_nearby_count(self,
                          player_pos: Tuple[int],
                          entities: List[KutuluEntity],
                          lines: List[List[str]],
                          limit: int):
        e_nearby_count = 0
        for e in entities:
            e_pos = (e.x, e.y)
            if distance(player_pos, e_pos) <= limit:
                path = find_path(player_pos, e_pos, lines)
                if len(path) <= limit:
                    e_nearby_count += 1
        return e_nearby_count
    
    def _get_min_dist(self,
                      player_pos: Tuple[int],
                      entities: List[KutuluEntity],
                      lines: List[List[str]],
                      limit: int):
        min_dist = 1000
        for e in entities:
            e_pos = (e.x, e.y)
            if distance(player_pos, e_pos) <= limit:
                path = find_path(player_pos, e_pos, lines)
                min_dist = min(min_dist, len(path))
        return min_dist

    def _score_moves_by_wanderers(self, player_pos, observation: AgentObservation, limit=4):
        wanderers = self._get_wanderers(observation)
        wanderers = [
            w.to_dict() for w in wanderers
            if distance(player_pos, (w.x, w.y)) <= limit
        ]
        all_distances = get_all_distances(wanderers, player_pos, observation.info.lines)
        all_distances = {
            # [1, 2, 2] -> 0.6385
            k: np.exp(-np.array(v)).sum()
            for k, v in all_distances.items()
        }
        move_scores = [all_distances.get(rel_pos, 0) for rel_pos in REL_POSITIONS[:4]]
        return move_scores

    def _compute_shaped_reward(self,
                               observation: AgentObservation,
                               action: int,
                               next_observations: AgentObservation,
                               original_reward: float,
                               other_rewards: List[float]):
        reward = 0
        if self.actions[action] == EffectType.PLAN.value:
            player = next_observations.obs.entities[0]
            assert player.id == next_observations.player_id
            player_pos = (player.x, player.y)
            explorers = self._get_explorers(next_observations, player.id)
            players_nearby_count = self._get_nearby_count(
                player_pos,
                explorers,
                next_observations.info.lines,
                PLAN_DISTANCE,
            )
            if players_nearby_count > 0:
                plan_bonus = self.good_plan_bonus * (players_nearby_count + 1)
            else:
                plan_bonus = self.bad_plan_bonus
            reward += plan_bonus
            if self.verbose:
                print(f"plan_bonus: {plan_bonus}")
        elif self.actions[action] == EffectType.LIGHT.value:
            player = observation.obs.entities[0]
            assert player.id == observation.player_id
            player_pos = (player.x, player.y)
            wanderers = self._get_wanderers(observation)
            is_bad_light = min([distance(player_pos, (w.x, w.y)) for w in wanderers], default=0) <= 1
            if is_bad_light:
                light_bonus = self.bad_light_bonus
            else:
                cur_enemies_min_dist = self._get_min_dist(
                    player_pos,
                    wanderers,
                    observation.info.lines,
                    LIGHT_DISTANCE,
                )
                next_enemies_min_dist = self._get_min_dist(
                    player_pos,
                    self._get_wanderers(next_observations),
                    next_observations.info.lines,
                    LIGHT_DISTANCE,
                )
                if cur_enemies_min_dist < next_enemies_min_dist:
                    light_bonus = self.good_light_bonus
                else:
                    light_bonus = self.bad_light_bonus
            reward += light_bonus
            if self.verbose:
                print(f"light_bonus: {light_bonus}")
        elif self.actions[action] == EffectType.YELL.value:
            yell_bonus = self.bad_yell_bonus
            other_rewards = [r for r in other_rewards if r is not None]
            if len(other_rewards) != 0:
                min_other_reward = min(other_rewards)
                if min_other_reward < 0:
                    player = observation.obs.entities[0]
                    assert player.id == observation.player_id
                    player_pos = (player.x, player.y)
                    explorers = self._get_explorers(observation, player.id)
                    players_nearby_count = self._get_nearby_count(
                        player_pos,
                        explorers,
                        observation.info.lines,
                        1,
                    )
                    if players_nearby_count > 0:
                        yell_bonus = - self.yell_bonus_coef * min_other_reward
            reward += yell_bonus
            if self.verbose:
                print(f"yell_bonus: {yell_bonus}")
        elif self.actions[action] == MoveType.WAIT.value:
            if original_reward < 0:
                wait_bonus = self.wait_reward_coef * original_reward
                reward += wait_bonus
                if self.verbose:
                    print(f"wait_bonus: {wait_bonus}")
        else:
            # MOVE
            assert self.actions[action] in (
                MoveType.UP.value,
                MoveType.RIGHT.value,
                MoveType.DOWN.value,
                MoveType.LEFT.value,
            )
            player = observation.obs.entities[0]
            assert player.id == observation.player_id
            player_pos = (player.x, player.y)
            move_scores = self._score_moves_by_wanderers(player_pos, observation)
            if move_scores[action] > 0 and move_scores[action] == max(move_scores):
                towards_enemy_bonus = self.bad_towards_enemy_bonus
                reward += towards_enemy_bonus
                if self.verbose:
                    print(f"towards_enemy_bonus: {towards_enemy_bonus}")
        if self.other_reward_coef is not None:
            other_rewards = [r for r in other_rewards if r is not None]
            if len(other_rewards) != 0:
                min_other_reward = min(other_rewards)
                min_other_reward = min(min_other_reward, 0)
                other_reward_bouns = - self.other_reward_coef * min_other_reward
                reward += other_reward_bouns
                if self.verbose:
                    print(f"other_reward_bouns: {other_reward_bouns}")
        if self.good_explorers_nearby_bonus != 0 or self.bad_explorers_nearby_bonus != 0:
            player = next_observations.obs.entities[0]
            assert player.id == next_observations.player_id
            player_pos = (player.x, player.y)
            explorers = self._get_explorers(next_observations, player.id)
            players_nearby_count = self._get_nearby_count(
                player_pos,
                explorers,
                next_observations.info.lines,
                PLAN_DISTANCE,
            )
            if players_nearby_count > 0:
                nearby_bonus = self.good_explorers_nearby_bonus * players_nearby_count
            else:
                nearby_bonus = self.bad_explorers_nearby_bonus
            reward += nearby_bonus
            if self.verbose:
                print(f"nearby_bonus: {nearby_bonus}")
        if self.shelter_bonus != 0:
            player = next_observations.obs.entities[0]
            assert player.id == next_observations.player_id
            lines = next_observations.info.lines
            player_pos = (player.x, player.y)
            shelters = self._get_shelters(next_observations)
            active_shelters = [sh for sh in shelters if sh.param0 > 0]
            shelters_underfoot_count = self._get_nearby_count(
                player_pos,
                active_shelters,
                next_observations.info.lines,
                0,
            )
            if shelters_underfoot_count > 0:
                reward += self.shelter_bonus
                if self.verbose:
                    print(f"shelter_bonus: {self.shelter_bonus}")

        return reward

    def recalculate_rewards(self,
                            rewards: List[float],
                            actions: List[int],
                            states: List,
                            dones: List[bool],
                            other_rewards: List[List[float]],
                            observations: List):
        assert dones[-1]
        assert sum(dones[:-1]) == 0
        rewards = list(rewards)
        T = len(rewards)
        for t in range(0, T - 1):
            rewards[t] += self._compute_shaped_reward(
                observations[t], actions[t], observations[t+1], rewards[t], other_rewards[t],
            )
        return rewards
