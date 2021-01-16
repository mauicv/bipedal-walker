from dataclasses import dataclass
import numpy as np


@dataclass
class Box:
    shape: int
    high: list
    low: list

    def sample(self):
        return np.clip(np.random.normal(0, 0.1, size=(self.shape)), 0, 1)
