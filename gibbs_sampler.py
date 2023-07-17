#%% Librairies

import numpy as np
from ising_model import ising_model

alpha = 0
beta = 0.2
N_iter=10**4
size=10

#%% gibbs sampler 

class gibbs_sampler:

    def __init__(self, N_iter, model):
        self.N_iter = N_iter
        self.model = model

    def sampling(self):
        for iter in range(self.N_iter):
            # for each point of the grid ...
            for k_i in range(self.model.size):
                for k_j in range(self.model.size): 
                    proba_1 = self.model.cond_prob(1,k_i, k_j)
                    # whether 1 or -1 at the next iteration
                    if np.random.uniform(0,1) < proba_1:
                        self.model.grid[k_i,k_j] = 1
                    else:
                        self.model.grid[k_i,k_j] = -1
        return self.model.grid
    
#%% Tests :

# model = ising_model(10)
# ising_sampler = gibbs_sampler(N_iter, model)
# print(ising_sampler.sampling())