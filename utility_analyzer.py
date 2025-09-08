from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Literal, Any, Type
import numpy as np

from solution.mdp import MDP, Action
from solution.policies import Policy

TerminalKind = Literal["goal", "hole", "none"]


@dataclass
class UtilityAnalyzer:
    mdp: MDP
    gamma: float = 0.99
    step_limit: int = 100

    def run_trial(
        self, policy_cls: Type[Policy], seed: int
    ) -> Tuple[float, int, TerminalKind]:
        """
        Instantiate a fresh policy with its own rng(seed) and simulate one episode.
        Returns (discounted_utility, length, terminal_kind).
        """
        rng = np.random.default_rng(seed)
        policy = policy_cls(self.mdp, rng)
        s = self.mdp.start_state()
        total, gamma_t, length = 0.0, 1.0, 0
        term_kind = "none"

        for t in range(self.step_limit):
            a = policy(s)
            ns, r = self.mdp.step(s, a, rng)
            total += gamma_t * r
            gamma_t *= self.gamma
            length += 1
            if self.mdp.is_terminal(ns):
                cell = self.mdp.grid[ns[0]][ns[1]] if ns != "⊥" else None
                if cell == "G": term_kind = "goal"
                elif cell == "H": term_kind = "hole"
                s = self.mdp.absorb
                break
            s = ns
        return total, length, term_kind

    def evaluate(
        self, policy_cls: Type[Policy], n_trials: int, base_seed: int = 0
    ) -> Dict[str, Any]:
        utils, lengths, terms = [], [], []
        for i in range(n_trials):
            u,l,t = self.run_trial(policy_cls, base_seed+i)
            utils.append(u); lengths.append(l); terms.append(t)
        return {
            "n_trials": n_trials,
            "mean_utility": float(np.mean(utils)),
            "utility_variance": float(np.var(utils)),
            "p_goal": terms.count("goal")/n_trials,
            "p_hole": terms.count("hole")/n_trials,
            "p_none": terms.count("none")/n_trials,
            "mean_length": float(np.mean(lengths)),
        }