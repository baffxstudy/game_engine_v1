import random
from typing import List, Dict

class MonteCarloSimulator:
    """
    Runs thousands of simulations to determine which 
    outcomes are 'Value' bets vs 'Trap' bets.
    """
    def __init__(self, iterations: int = 10000):
        self.iterations = iterations

    def simulate_match(self, true_prob: float) -> float:
        """Returns the simulated success rate of the selection."""
        wins = sum(1 for _ in range(self.iterations) if random.random() < true_prob)
        return wins / self.iterations