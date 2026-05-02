import gymnasium as gym
import numpy as np

class FrozenLakeMaskWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.desc = self.unwrapped.desc
        self.nrow, self.ncol = self.desc.shape

    def action_masks(self, state=None):
        if state is None:
            state = self.unwrapped.s
            
        row, col = state // self.ncol, state % self.ncol
        valid_actions = np.ones(4, dtype=np.int8)

        if col > 0 and self.desc[row, col - 1] == b'H':
            valid_actions[0] = 0
        if row < self.nrow - 1 and self.desc[row + 1, col] == b'H':
            valid_actions[1] = 0
        if col < self.ncol - 1 and self.desc[row, col + 1] == b'H':
            valid_actions[2] = 0
        if row > 0 and self.desc[row - 1, col] == b'H':
            valid_actions[3] = 0

        if not valid_actions.any():
            valid_actions = np.ones(4, dtype=np.int8)

        return valid_actions