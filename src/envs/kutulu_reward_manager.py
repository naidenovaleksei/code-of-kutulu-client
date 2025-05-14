
DEFAULT_REWARD_FOR_WIN = +10
DEFAULT_REWARD_FOR_LOSE = -5
DEFAULT_STEP_BONUS = +0.1
# DEFAULT_SPOOKED_BONUS = -1

DEFAULT_SANITY_COEFFICIENT = 1 / 20
# reward_for_lose = -5 (DEFAULT_REWARD_FOR_LOSE)
# reward_for_lose = 10 (DEFAULT_REWARD_FOR_WIN)
# reward_for_alive = 0.1 (DEFAULT_STEP_BONUS)
# reward_for_not_alone = [0..0.4] ([0..5] * DEFAULT_SANITY_COEFFICIENT(1/20))
# reward_for_spooked = -1 (MINION_SANITY_LOST(-20) * DEFAULT_SANITY_COEFFICIENT(1/20))

class KutuluRewardManager:
    def __init__(self,
                spread_madness_per_turn,
                reward_for_win=DEFAULT_REWARD_FOR_WIN,
                reward_for_lose=DEFAULT_REWARD_FOR_LOSE,
                step_bonus=DEFAULT_STEP_BONUS,
                sanity_coef=DEFAULT_SANITY_COEFFICIENT,
                **kwargs):
        self.spread_madness_per_turn = spread_madness_per_turn        
        self.reward_for_win = reward_for_win
        self.reward_for_lose = reward_for_lose
        self.step_bonus = step_bonus
        self.sanity_coef = sanity_coef

        self.players = {}
        self.death_turns = {}

    def update_players(self, players):
        self.players = players

    def calculate_rewards(self, entities, turn, game_over):
        active_sanity = {
            e['id']: e['param0']
            for e in entities
            if e['kind'] == 'EXPLORER'
        }

        rewards = {}
        for player_id, player in self.players.items():
            if player_id in active_sanity:
                sanity_loss = active_sanity[player_id] - player['sanity']
                sanity_loss_corrected = sanity_loss + self.spread_madness_per_turn
                sanity_loss_normalized = sanity_loss_corrected * self.sanity_coef
                rewards[player_id] = sanity_loss_normalized + self.step_bonus
                if game_over and self.reward_for_win is not None:
                    rewards[player_id] += self.reward_for_win
            else:
                if player_id not in self.death_turns:
                    self.death_turns[player_id] = turn
                    if not game_over and self.reward_for_lose is not None:
                        rewards[player_id] = self.reward_for_lose

        if game_over and len(active_sanity) == 0:
            max_death_turns = max(self.death_turns.values())
            for player_id, player in self.players.items():
                if (self.death_turns[player_id] == max_death_turns) \
                    and self.reward_for_win is not None:
                    rewards[player_id] = rewards.get(player_id, 0) + self.reward_for_win

        reward = [rewards.get(player_id) for player_id in self.players]
        return reward
