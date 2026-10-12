import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        gamma = np.array(gamma)
        x = np.array(x)
        mean = np.mean(x ** 2)
        RMS = np.sqrt(mean + eps)
        res = x / RMS 
        return np.round(res * gamma, 4).tolist()
