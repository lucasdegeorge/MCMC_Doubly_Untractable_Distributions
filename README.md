# Markov Chain Monte Carlo methods project

This project is based on the first numerical experiement of this [paper](https://doi.org/10.1214/15-STS523). It aims at recovering the posterior doubly-intractable distribution of a parameter

* First step: we implement a Gibbs sampler to sample from the Ising model.

Considering a $N\times N$ grid of spins : $\textbf{y} = (y_1,\dots,y_{N^2})$ where each $y_i \in \{-1,+1\}. $, The likelihood of the Ising model is defined by : 

$$ p(\textbf{y} ; \alpha, \beta) = \frac{1}{\mathcal{Z}(\alpha, \beta)} \exp \left( \alpha \sum_{i=1}^{N^2} y_i + \beta \sum_{(i,j) \in V} y_iy_j  \right) $$

* Second step: we implement an exchange algorithm to sample from the posterior distribution. More details about the exchange algorithm by Murray et al. [here](https://arxiv.org/abs/1206.6848)

Given a grid of spins $\textbf{y} = (y_1,\dots,y_{N^2})$, we want to recover the parameters $\alpha$ and $\beta$ which led to the generation of $\textbf{y}$. Assuming $\alpha = 0$, the posterior probability we want to infer is 

$$ p(\beta | \textbf{y}) = \frac{p(\textbf{y}|\beta) \times p(\beta)}{p(\textbf{y})} = \frac{f(\textbf{y}|\beta)}{\mathcal{Z}(\beta)}\times p(\beta) \times \frac{1}{p(\textbf{y})} $$

where $\mathcal{Z}(\beta)$ and $p(\textbf{y})$ are intractable.

We can't use the Metropolis Hasting algorithm as $\mathcal{Z}(\beta)$ is untractable. So, we use and implement the Exchange algorithm. 

* Results: (more details on the project render presentation)

We run the algorithm with $T = 10^4$, $N_{gs} = 10^3$ and $\sigma = 0.1$. 

<div align="center">
<kbd><img src="https://github.com/lucasdegeorge/lucasdegeorge/blob/main/long%20run.png" width="450" height="350" /></kbd>
</div>
The algorithm seems to converge towards 0.2 (the value used by the Gibbs sampler. 
