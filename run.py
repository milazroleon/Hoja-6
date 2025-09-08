from __future__ import annotations
import json

from solution.lake_mdp import LakeMDP
from solution.policies import RandomPolicy, CustomPolicy
from solution.utility_analyzer import UtilityAnalyzer

# Default map (4x4 from the assignment PDF)
DEFAULT_MAP = [
    ["S", "F", "F", "F"],
    ["F", "H", "F", "H"],
    ["F", "F", "F", "H"],
    ["H", "F", "F", "G"],
]

def evaluate_all(trials: int = 100, base_seed: int = 123):
    """
    Evaluate RandomPolicy and CustomPolicy for γ ∈ {0.5, 0.9, 1.0}.
    Returns a JSON-serializable dict with summaries and the winner per γ.
    """
    report = {"n_trials": int(trials), "base_seed": int(base_seed), "gammas": {}}

    for gamma in gammas:
        mdp = LakeMDP(DEFAULT_MAP)
        ua = UtilityAnalyzer(mdp, gamma=gamma)

        sum_rand = ua.evaluate(RandomPolicy, trials, base_seed)
        sum_cust = ua.evaluate(CustomPolicy, trials, base_seed)

        # elegir mejor
        if sum_cust["mean_utility"] > sum_rand["mean_utility"]:
            best = "custom"
        elif sum_cust["mean_utility"] < sum_rand["mean_utility"]:
            best = "random"
        else:  # empate
            best = "custom" if sum_cust["utility_variance"] < sum_rand["utility_variance"] else "random"


    report["gammas"][str(gamma)] = {
        "random": sum_rand,
        "custom": sum_cust,
        "winner": best,
    }

    return report