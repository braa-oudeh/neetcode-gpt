import numpy as np
from typing import Tuple, List

class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:

        x_hat = None
        m = 0.1

        batch_size = len(x)
        features_size = len(x[0])
        epsilon = 1e-5
        mub= []
        var = []
        features = []

        if training == True: 
            for i in range(features_size): 
                inner_list = []
                for j in range(batch_size): 
                    inner_list.append(x[j][i])
                features.append(inner_list)
            for feature in features:
                mu = np.mean(feature) 
                mub.append(np.mean(feature))
                var.append(np.mean((feature - mu) ** 2) ) 
            var_np = np.array(var)
            x_hat = (np.array(x) - mub) / np.sqrt(var_np + epsilon)
            
            running_mean = (1 - m) * np.array(running_mean) + (m * np.array(mub))
            running_var = (1 - m) * np.array(running_var) + (m * var_np)     
        else: 
            x_hat = (x - np.array(running_mean)) / (np.sqrt(np.array(running_var) + epsilon))
        y = np.round(gamma * x_hat + beta,4)
        return (y, np.round(running_mean,4), np.round(running_var,4))
