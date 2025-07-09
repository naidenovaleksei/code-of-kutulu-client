import math
from collections import defaultdict
from itertools import combinations

class EloLeague:
    def __init__(self, default_rating=1000, k=32):
        self.default_rating = default_rating
        self.ratings = defaultdict(self._get_default_rating)
        self.k = k
        self.games_played = defaultdict(int)
        self.wins = defaultdict(int)
        self.total_rank = defaultdict(int)

    def _get_default_rating(self):
        return self.default_rating

    def expected(self, Ra, Rb):
        return 1 / (1 + 10 ** ((Rb - Ra) / 400))

    def update_elo_pair(self, Ra, Rb, Sa):
        Ea = self.expected(Ra, Rb)
        delta = self.k * (Sa - Ea)
        return Ra + delta, Rb - delta

    def record_match(self, ranked_players):
        """
        :param ranked_players: список кортежей (player_id, score), отсортированных по месту (чем выше, тем лучше)
                               Пример: [('A', 100), ('B', 80), ('C', 30)] — A победил, C проиграл
        """
        ranked_players = sorted(ranked_players, key=lambda x: x[1], reverse=True)
        # Обновление рейтингов по pairwise матчам
        for i in range(len(ranked_players)):
            for j in range(i + 1, len(ranked_players)):
                player_i, _ = ranked_players[i]
                player_j, _ = ranked_players[j]

                Ri = self.ratings[player_i]
                Rj = self.ratings[player_j]

                # i победил над j
                Ri_new, Rj_new = self.update_elo_pair(Ri, Rj, 1)
                self.ratings[player_i] = Ri_new
                self.ratings[player_j] = Rj_new

                self.wins[player_i] += 1
                self.games_played[player_i] += 1
                self.games_played[player_j] += 1

        # Обновим средние позиции
        for rank, (pid, _) in enumerate(ranked_players):
            self.total_rank[pid] += rank

    def get_rating(self, player_id):
        return self.ratings[player_id]

    def get_stats(self, player_id):
        games = self.games_played[player_id]
        wins = self.wins[player_id]
        avg_rank = self.total_rank[player_id] / games if games else None
        return {
            "rating": self.ratings[player_id],
            "games_played": games,
            "wins": wins,
            "avg_rank": avg_rank,
        }

    def leaderboard(self, top_n=10):
        return sorted(self.ratings.items(), key=lambda x: -x[1])[:top_n]
