import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        # Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)
        # Normalize x, then scale by gamma
        # Return result rounded to 4 decimal places as a list
        gamma = np.array(gamma)
        x = np.array(x)
        mean = np.mean(x ** 2)
        RMS = np.sqrt(mean + eps)
        res = x / RMS 
        return np.round(res * gamma, 4).tolist()
