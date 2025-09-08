from __future__ import annotations

from solution.policy import Policy
from solution.lake_mdp import DOWN, RIGHT, Action, State


class RandomPolicy(Policy):
    """Uniform over legal actions."""

    def _decision(self, s: State) -> Action:
        acts = list(self.mdp.actions(s))
        return self.rng.choice(acts)


class CustomPolicy(Policy):
    """
    Simple deterministic rule that avoids an immediate hole:
      - Prefer DOWN if the cell below is NOT a hole.
      - Else prefer RIGHT if the cell to the right is NOT a hole.
      - Else pick the first legal action (covers absorbing ⊥ case, or being boxed in).
    Note: This checks intended cells only (not slip outcomes).
    """

    def _decision(self, s: State) -> Action:
        acts = list(self.mdp.actions(s))
        if s == "⊥": return "⊥"
        r,c = s

        if r+1 < self.mdp.rows and self.mdp.grid[r+1][c] != "H":
            return "DOWN"

        if c+1 < self.mdp.cols and self.mdp.grid[r][c+1] != "H":
            return "RIGHT"
        return acts[0]
