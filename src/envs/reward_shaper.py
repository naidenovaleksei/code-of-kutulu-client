from typing import List, Tuple

import numpy as np
import torch.nn.functional as F
from collections import deque

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

METRICS_SMOOTH_COEF = 0.05
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
        return np.exp(-np.array(explorers_distances)).sum()

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
        return np.exp(-min(shelters_distances))

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
        if original_reward < 0 and self.actions[action] not in (
            MoveType.UP.value,
            MoveType.RIGHT.value,
            MoveType.DOWN.value,
            MoveType.LEFT.value,
        ):
            no_move_bonus = self.no_move_reward_coef * original_reward
            reward += no_move_bonus
            if self.verbose:
                print(f"no_move_bonus: {no_move_bonus}")
            return reward

        wanderers_nearby_bonus = self._wanderers_nearby_bonus(observation, next_observations)
        reward += wanderers_nearby_bonus
        if self.verbose:
            print(f"wanderers_nearby_bonus: {wanderers_nearby_bonus}")

        explorers_nearby_bonus = self._explorers_nearby_bonus(observation, next_observations)
        reward += explorers_nearby_bonus
        if self.verbose:
            print(f"explorers_nearby_bonus: {explorers_nearby_bonus}")

        shelters_nearby_bonus = self._shelters_nearby_bonus(observation, next_observations)
        reward += shelters_nearby_bonus
        if self.verbose:
            print(f"shelters_nearby_bonus: {shelters_nearby_bonus}")

        others_sanity_loss_bonus = self._others_sanity_loss_bonus(observation, next_observations)
        reward += others_sanity_loss_bonus
        if self.verbose:
            print(f"others_sanity_loss_bonus: {others_sanity_loss_bonus}")

        if self.actions[action] == EffectType.PLAN.value:
            assert original_reward >= 0
            plan_bonus = self._plan_bonus(observation, next_observations)
            reward += plan_bonus
            if self.verbose:
                print(f"plan_bonus: {plan_bonus}")
        elif self.actions[action] == EffectType.LIGHT.value:
            assert original_reward >= 0
            light_bonus = self._light_bonus(observation, next_observations)
            reward += light_bonus
            if self.verbose:
                print(f"light_bonus: {light_bonus}")
        elif self.actions[action] == EffectType.YELL.value:
            assert original_reward >= 0
            yell_bonus = self._yell_bonus(observation, next_observations)
            reward += yell_bonus
            if self.verbose:
                print(f"yell_bonus: {yell_bonus}")
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
