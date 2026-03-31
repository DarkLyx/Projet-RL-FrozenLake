import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class SB3MetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rewards_history = []
        self.lengths_history = []
        self.current_length = 0

    def _on_step(self) -> bool:
        self.current_length += 1
        if self.locals["dones"][0]:
            reward = self.locals["rewards"][0]
            self.rewards_history.append(1 if reward > 0 else 0)
            self.lengths_history.append(self.current_length)
            self.current_length = 0
        return True