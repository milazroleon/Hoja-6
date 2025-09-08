from __future__ import annotations
from typing import Iterable, List, Tuple

from solution.mdp import MDP, State, Action

UP, RIGHT, DOWN, LEFT, ABSORB = "UP", "RIGHT", "DOWN", "LEFT", "⊥"


class LakeMDP(MDP):
    """
    Grid map (matrix of single-character strings), e.g.:
      [
        ['S','F','F','F'],
        ['F','H','F','F'],
        ['F','F','F','F'],
        ['H','F','F','G'],
      ]

    Rewards are *state-entry* rewards. After entering H or G, the next state is
    the absorbing state ⊥ with only legal action ⊥ and 0 reward forever.
    """

    def __init__(self, grid: Iterable[Iterable[str]]):
        self.grid = [list(row) for row in grid]
        self.rows = len(self.grid)
        self.cols = len(self.grid[0])

        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == "S":
                    self.start = ((r, c), "S")

        self.absorb = (ABSORB, ABSORB)

    # --- MDP interface -----------------------------------------------------
    def start_state(self) -> State:
        return self.start

    def actions(self, s: State) -> Iterable[Action]:
        if s == self.absorb:
            return [ABSORB]
        (r, c), val = s
        if val in ("H", "G"):
            return [ABSORB]
        return [UP, RIGHT, DOWN, LEFT]

    def reward(self, s: State) -> float:
        if s == self.absorb:
            return 0.0
        (r, c), val = s
        if val == "F":
            return 0.1
        if val == "H":
            return -1.0
        if val == "G":
            return 1.0
        return 0.0


    def is_terminal(self, s: State) -> bool:
        if s == self.absorb:
            return True
        (r, c), val = s
        return val in ("H", "G")

    def transition(self, s: State, a: Action) -> List[Tuple[State, float]]:
        if s == self.absorb:
            return [(self.absorb, 1.0)]
        if self.is_terminal(s):
            return [(self.absorb, 1.0)]


        moves = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
        lateral = {
            "UP": ["LEFT", "RIGHT"],
            "DOWN": ["LEFT", "RIGHT"],
            "LEFT": ["UP", "DOWN"],
            "RIGHT": ["UP", "DOWN"],
        }

        (r, c), _ = s
        dist = []
        for act, prob in [(a, 0.8)] + [(lat, 0.1) for lat in lateral[a]]:
            ns = self._move((r, c), moves[act])
            rr, cc = ns
            sym = self.grid[rr][cc]
            dist.append((((rr, cc), sym), prob))
        return dist

    def _move(self, pos: Tuple[int, int], delta: Tuple[int, int]) -> Tuple[int, int]:
        r, c = pos
        nr, nc = r + delta[0], c + delta[1]
        if 0 <= nr < self.rows and 0 <= nc < self.cols:
            return (nr, nc)
        return (r, c)
