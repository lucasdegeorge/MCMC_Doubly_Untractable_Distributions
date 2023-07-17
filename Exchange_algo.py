#%% Librairies

import numpy as np
import matplotlib.pyplot as plt
from ising_model import ising_model
from gibbs_sampler import gibbs_sampler

T = 1000
variance = 0.08
N_iter_gs = 10**3 # nb of iterations in the gibbs sampler

def gaussian_kernel(x, var=variance, data=None):
    return np.random.normal(x, var)

def gaussian_kernel_density(x, mu, var=variance):
    return (1/np.sqrt(2*np.pi*var)) * np.exp(-((x-mu)**2)/2*var)

def uni(x):
    return 1 

#%% exchange algorithm

class exchange_algorithm:

    def __init__(self, T, kernel, kernel_density, prior):
        self.T = T
        self.kernel = kernel # Q in the notation of the paper MCMC for doubly-intractable distributions
        self.kernel_density = kernel_density # without the normalization factor | q in the notation of the paper MCMC for doubly-intractable distributions
        self.prior = prior 

    def execution(self, data, initial=None): # data is the y with the notation of the paper MCMC for doubly-intractable distributions
        #Initialization
        if initial is not None:
            current_theta = initial
        else:
            current_theta = np.random.uniform(0,1)

        theta_samples = np.zeros(self.T)
        a_rates = np.zeros(self.T)
        a_bounded_rates = np.zeros(self.T)

        for iter in range(self.T):
            # line 2
            theta_prime = self.kernel(current_theta)
            if theta_prime > 1:
                theta_prime = 1
            if theta_prime < 0:
                theta_prime = 0
            # line 3
            aux_model = ising_model(10, 0, theta_prime, None)
            sampler = gibbs_sampler(N_iter_gs, aux_model)
            sampler.sampling()
            # line 4
            frac_prior = 1 # = self.prior(theta_prime) / self.prior(current_theta)
            frac_kernel = self.kernel_density(current_theta, theta_prime) / self.kernel_density(theta_prime, current_theta)
            frac_data = data.likelihood(theta_prime) / data.likelihood(current_theta)
            frac_aux = aux_model.likelihood(current_theta) / aux_model.likelihood(theta_prime)
            a = frac_kernel * frac_prior * frac_data * frac_aux
            a_bounded = max(min(a,1),0)
            # line 5 and 6
            if np.random.uniform(0,1) < a:
                current_theta = theta_prime

            theta_samples[iter] = current_theta
            a_rates[iter] = a
            a_bounded_rates[iter] = a_bounded

            # print each 1000 iterations
            if iter % 99 == 0:
                S = [(1/i)*np.sum(theta_samples[:i]) for i in range(1,iter)]
                plt.figure()
                plt.plot([i for i in range(len(S))], S)
                plt.show()
            
        return theta_samples, a_rates, a_bounded_rates
    
    def results(self, data, initial=None):
        theta_samples, a_rates, a_bounded_rates = self.execution(data, initial)
        print("----------------------------")
        print("------Results on theta------")
        print("Mean:", np.mean(theta_samples))
        print("Standard deviation:", np.std(theta_samples))
        print("Initial theta:", theta_samples[0])
        print("----------------------------")
        print("--------Results on a--------")
        print("Mean:", np.mean(a_rates))
        print("Standard deviation:", np.std(a_rates))
        print("----------------------------")
        print("----Results on a_bounded----")
        print("Mean:", np.mean(a_bounded_rates))
        print("Standard deviation:", np.std(a_bounded_rates))
        


#%% Tests

model_y = ising_model(10)
exchange_algo = exchange_algorithm(T, gaussian_kernel, gaussian_kernel_density, uni)
exchange_algo.results(model_y)
