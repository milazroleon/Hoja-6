from __future__ import annotations
from typing import Iterable, List, Tuple, Dict

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
        self.rows = len(grid)
        self.cols = len(grid[0])

        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == "S":
                    self.start = (r, c)
        self.absorb = "⊥" 

    # --- MDP interface -----------------------------------------------------
    def start_state(self) -> State:
        return self.start

    def actions(self, s: State) -> Iterable[Action]:
        return [UP, RIGHT, DOWN, LEFT] if s != self.absorb else [ABSORB]

    def reward(self, s: State) -> float:
        if self.grid[s[0]][s[1]] == "H":
            return -1.0
        elif self.grid[s[0]][s[1]] == "G":
            return 1.0
        elif self.grid[s[0]][s[1]] in ("F", "F"):
            return 1.0
        else:
            return 0.0

    def is_terminal(self, s: State) -> bool:
        return s == self.absorb

    def transition(self, s: State, a: Action) -> List[Tuple[State, float]]:
        if s == self.absorb:
            return [(self.absorb, 1.0)]

        if self.is_terminal(s):
            return [(self.absorb, 1.0)]

        moves = {"UP": (-1,0), "DOWN": (1,0), "LEFT": (0,-1), "RIGHT": (0,1)}
        lateral = {"UP": ["LEFT","RIGHT"], "DOWN": ["LEFT","RIGHT"],
                   "LEFT": ["UP","DOWN"], "RIGHT": ["UP","DOWN"]}

        dist = []
        for act, prob in [(a,0.8)] + [(lat,0.1) for lat in lateral[a]]:
            ns = self._move(s, moves.get(act,(0,0)))
            dist.append((ns, prob))
        return dist
    
    def _move(self, s, delta):
        r, c = s
        nr, nc = r + delta[0], c + delta[1]
        if 0 <= nr < self.rows and 0 <= nc < self.cols:
            return (nr,nc)
        return s 