from dataclasses import dataclass
import numpy as np


@dataclass
class Box:
    dim: int
    high: list
    low: list

    def sample(self):
        return np.clip(np.random.normal(0, 0.1, size=(self.dim)), 0, 1)
