# Author: Simon Guldager Andersen
# Date (latest update): 

### SETUP ------------------------------------------------------------------------------------

## Imports:
import os, sys
import emcee
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats, integrate, interpolate, optimize
from iminuit import Minuit
from matplotlib import rcParams
from cycler import cycler

sys.path.append('Appstat2022\\External_Functions')
from ExternalFunctions import Chi2Regression, BinnedLH, UnbinnedLH
from ExternalFunctions import nice_string_output, add_text_to_ax    # Useful functions to print fit results on figure


## Change directory to current one
os.chdir('AdvAppStat\\MCMC')


### FUNCTIONS ----------------------------------------------------------------------------------

def Metropolis_1param(prop_distribution, proposal_distribution, Nsamples, \
    Nburnin, boundaries, starting_point = None):
    """
    Assuming only 1 parameter
    """
    ## Intialize
    accepted_samples = np.empty(Nsamples)
    Naccepted_jumps = 0
    iterations = 0

    ## If starting point is not provided, gen. a random one
    if starting_point is None:
        x_old = boundaries[0] + np.random.rand() * boundaries[1]
    else:
        x_old = starting_point

    while iterations - Nburnin < Nsamples:
        ## Propose a jump
        x_new = proposal_distribution(x_old)

        ## Enforce periodic boundaries
        if x_new > boundaries[1]:
            x_new = max(boundaries[0], boundaries[0] + (x_new - boundaries[1]))
        elif x_new < boundaries[0]:
            x_new = min(boundaries[1], boundaries[1] + (x_new - boundaries[0]))

        ## Accept new step with transition probability p_accept
        prob_fraction = (prop_distribution(x_new) * proposal_distribution(x_old, x_new)) \
            / (prop_distribution(x_old) * proposal_distribution(x_new, x_old))

        p_accept = min(1, prob_fraction)
        if np.random.rand() < p_accept:
            x_old = x_new.astype('float')
            if iterations >= Nburnin:
                Naccepted_jumps +=1

        # Collect values after the burn in phase
        if iterations >= Nburnin:
            accepted_samples[iterations - Nburnin] = x_old

        iterations += 1

    # Calc acceptance_rate
    acceptance_rate = Naccepted_jumps / (Nsamples - Nburnin)

    return accepted_samples, acceptance_rate



### MAIN ---------------------------------------------------------------------------------------

def EXC1_and_EXC3_4():
       ## Use Bayesian framework to calc prop of coming of heads. prior is beta distribution (p,alpha,beta), with p
    # prop of coming up heads. Likelihood func is binom func Binom(k,n,p)
    # alpha, beta, k, n are fixed, and for each p, we want to calc Post(p,k,n,alpha,beta) = binom(k,n,p)*Beta(p,alpha,beta)
    # Normalize functions
    # Plot against p for the given values

    alpha, beta = 5, 17
    multiplier = 1
    k, n = 66 * multiplier, 100 * multiplier

    prior = lambda p: stats.beta.pdf(p, a = alpha, b = beta)
    likelihood = lambda p: stats.binom.pmf(k = k, n = n, p = p)
    
  
    posterior = lambda p: likelihood(p) * prior(p)

    A_likelihood = integrate.quad(likelihood, 0, 1)[0]
    likelihood_normed = lambda p: likelihood(p) / A_likelihood

    A_posterior = integrate.quad(posterior, 0 , 1)[0]
    posterior_normed = lambda p: posterior(p) / A_posterior
   
    ## Check normalization
    Npoints = 1000
    p_range = np.linspace(0, 1, Npoints)
    dp = (1 - 0) * ( Npoints)
    dp = 1

    prior_vals = prior(p_range) 
    likelihood_vals = likelihood_normed(p_range) 
    post_vals = posterior_normed(p_range)
    
    max_post = np.argmax(post_vals)
    fig,ax = plt.subplots()

    ax.set(xlabel = 'Probability of heads', ylabel = 'Distribution value')
    ax.plot(p_range, post_vals , label = 'Posterior')
    ax.plot(p_range, prior_vals , label = 'Prior')  
    ax.plot(p_range, likelihood_vals , label = 'Likelihood')
    ax.plot([p_range[max_post], p_range[max_post]], [0, post_vals[max_post]], label = f'p = {p_range[max_post]:.2f}')
 
    ax.legend()

    ## EXC 3:
    # Redo exc1, this time using a Metropolis Hastings alg. to sample posterior
    # Sample points with acceptance prob min(1, (post(x_new) * prop_dist(x_old|x_new))/(post(x_old) * prop_dist(x_new|x_old)))
    # use prop_dist(x_new) = x_old + N(0, std = 0.3)
    # enforce limits properly

 
    std = .1

    def proposal_distribution(x, x_new_given_x = None):
        if x_new_given_x is None:
            return stats.norm.rvs(loc = x, scale = std)
        else:
            return stats.norm.pdf(x_new_given_x, loc = x, scale = std)

    ## Sample some posterior values
    Nsamples = 5000
    boundaries = [0.0, 1.0]
    Nburnin = 50

    sample_vals, acc_rate = Metropolis_1param(posterior_normed, proposal_distribution, Nsamples, Nburnin, boundaries)

    print("MC accep. rate: ", acc_rate)
    print("MC samples mean value: ", sample_vals.mean())

    fig0, ax0 = plt.subplots()
    iterations = np.arange(Nsamples)
    ax0.plot(iterations, sample_vals)
    ax0.set(xlabel = 'Iterations', ylabel = 'Sample value')
    
    fig2, ax2 = plt.subplots()
 
    bins = int(Nsamples/ 50)
    binwidth = (boundaries[1] - boundaries[0]) / bins
    ## plot scaled versions of pdfs
    posterior_vals_scaled = Nsamples * binwidth * posterior_normed(p_range)
    ax2.plot(p_range, posterior_vals_scaled, label = 'Posterior distribution')
    ax2.hist(sample_vals, range = boundaries, bins = bins, histtype = 'stepfilled', \
        alpha = .3, label = 'Homegrown MC samples')
    ax2.set(xlabel = 'Probability of heads', ylabel = 'Count')
  

    ## EXC 4: Redo exc 3 but with an external MCMC package
    def log_post(p, lower_bound = - 10_000):
        val = np.log(posterior(p))
        if np.isnan(val):
            return lower_bound
        else:
            return val

    sampler = emcee.EnsembleSampler(nwalkers = 2, ndim = 1, log_prob_fn = log_post)

    ## How to run a burn in phase of Nburnin, starting a p0, and then resetting so we can start from there
    p0 = np.random.rand(2,1)
    state = sampler.run_mcmc(p0, Nburnin)
    sampler.reset()

    ## Calc Nsamples MC samples
    sampler.run_mcmc(state, int(Nsamples/2))

    
    autocorr_values = sampler.get_autocorr_time() # Has length Nsamples
    dist_between_uncorr_samples = np.mean(autocorr_values)
    print("Est. autocorrelation time: ", dist_between_uncorr_samples)

    ## Get samples, taking only every N_jump_between_values steps (to get rid of correlation)
    ## the thin param determines no. of jumps between included samples
    samples = sampler.get_chain(flat = True, thin = 1) #dist_between_uncorr_samples)

    ax2.hist(samples, range = boundaries, bins = bins, histtype = 'stepfilled', \
        alpha = .3, label = 'emcee MC samples')

    ax2.legend()
    plt.show()

def EXC2():
    ## MCMC chains. Make a simulation of 2 diff chains starting at 100 and -27.
    ## The transition probability is governed by normal dist N(mu = x_t, sigma^2 = 1), where x_t is the
    # 'th iteration
    # x_t+1 is obtained by sampling 1 value of the distribution.
    # Plot the values vs the iterations number and observe the convergence.


    initial_vals = [-27, 100]
    std = 1
    sample_vals_left = []
    sample_vals_right = []

    x_old = np.array([initial_vals[0], initial_vals[1]], dtype = 'float')

    tolerance = 5e-1
    max_it = 1000
    iterations = 0

    while np.abs(x_old[0] - x_old[1]) > tolerance and iterations < max_it:
        iterations +=1
   
        # Gen new values
        x_new = stats.norm.rvs(0.5 * x_old, std ** 2)

        # Collect values
        sample_vals_left.append(x_old[0])
        sample_vals_right.append(x_old[1])

        # Uppdate old value
        x_old = x_new.astype('float')
        

    # Plot values against iteration no. for each chain series
    fig, ax = plt.subplots()
    Npoints = len(sample_vals_left)
    iterations = np.arange(Npoints)
 
    ax.set(xlabel = 'Iteration number', ylabel = 'Sample value')
    ax.plot(iterations, sample_vals_left, '.-', label = f'Chain starting at {initial_vals[0]}')
    ax.plot(iterations, sample_vals_right, '.-', label = f'Chain starting at {initial_vals[1]}')

    ax.legend()
    plt.show()

def EXC4():
    ## 
    pass



# Set plotting style
sns.set_theme()
sns.set_style("darkgrid")
sns.set_context("paper") #Possible are paper, notebook, talk and poster
rcParams['lines.linewidth'] = 2 
rcParams['axes.titlesize'] =  18
rcParams['axes.labelsize'] =  18
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['legend.fontsize'] = 15
rcParams['font.family'] = 'serif'
rcParams['figure.figsize'] = (9,6)
rcParams['axes.prop_cycle'] = cycler(color = ['teal', 'navy', 'coral', 'plum', 'purple', 'olivedrab',\
         'black', 'red', 'cyan', 'yellow', 'khaki','lightblue'])
np.set_printoptions(precision = 5, suppress=1e-10)

def main():

    exc1_and_exc3_4, exc2, exc4 = True, False, False

    if exc1_and_exc3_4:
        EXC1_and_EXC3_4()
    if exc2:
        EXC2()
    if exc4:
        EXC4()

 

if __name__ == '__main__':
    main()
