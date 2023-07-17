#%% Libraries and functions
import numpy as np

def zero_to_minus1(x):
    if x == 0: return -1 
    else: return x

def logistic(x,beta_0,beta_1):
    e = np.exp(beta_0 + beta_1*x)
    return e/(1+e)

#%% class

class ising_model:

    def __init__(self, size, alpha=0, beta=0.2, initial_state = None):
        self.size = size
        self.alpha = alpha
        self.beta = beta
        if initial_state is not None:
            self.grid = initial_state
        else:
            grid = np.random.randint(2, size=(self.size,self.size)) # initialize the matrix with 0 and 1
            self.grid = np.vectorize(zero_to_minus1)(grid) # replace all the O by -1

    def neighbors(self, i, j):
        if i==0:
            if j==0: return [(i+1,j),(i,j+1),(i,self.size-1), (self.size-1,j)]
            elif j==self.size-1: return [(i,j-1),(i+1,j),(i,0), (self.size-1, self.size-1)] 
            else: return [(i,j-1),(i+1,j),(i,j+1), (self.size-1,j)]
        elif i==self.size-1:
            if j==0: return [(i-1,j),(i,j+1),(self.size-1,self.size-1),(0,j)]
            elif j==self.size-1: return [(i,j-1),(i-1,j),(0,self.size-1),(i,0)] 
            else: return [(i,j-1),(i-1,j),(i,j+1),(0,j)]
        else:
            if j==0: return [(i-1,j),(i,j+1),(i+1,j),(i,self.size-1)]
            elif j==self.size-1: return [(i-1,j),(i,j-1),(i+1,j),(i,0)] 
            else: return [(i,j-1),(i-1,j),(i,j+1),(i+1,j)]

    def sum_neighbors(self, i, j):
        res = 0 
        for nei in self.neighbors(i,j):
            res += self.grid[nei]
        return res

    def following_neighbors(self, i, j):
        res = []
        for pair in self.neighbors(i,j):
            if pair[0] <= pair[1]:
                res.append(pair)
        return res
    
    def sum_following_neighbors(self,i,j):
        res = 0
        for nei in self.following_neighbors(i,j):
            res += self.grid[nei]
        return res
    
    def cond_prob(self, x, i, j):
        # this is the logisitic probability
        beta_0 = 0
        beta_1 = 2*(self.alpha + self.beta*self.sum_neighbors(i,j))
        return logistic(x, beta_0, beta_1)
    
    def likelihood(self, theta):
        # Warning: beta in the formula is replaced by theta in ordre tu compute the likelihood in the exchange algorithm
        sum_nei = 0 # compute the sum over all the pairs of neighbors
        for k_i in range(self.size):
            for k_j in range(self.size):
                sum_nei += self.sum_following_neighbors(k_i,k_j)
        return np.exp(self.alpha*np.sum(self.grid) + theta*sum_nei)


#%% Tests

# model = ising_model(0)
# print(model.grid)
# print(model.likelihood(0.2))
# # %%
