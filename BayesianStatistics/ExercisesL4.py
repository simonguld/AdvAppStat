# Author: Simon Guldager Andersen
# Date (latest update): 

### SETUP ------------------------------------------------------------------------------------

## Imports:
import os, sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib import rcParams
from cycler import cycler
from scipy.special import factorial

## Change directory to current one
os.chdir('...')


### FUNCTIONS ----------------------------------------------------------------------------------

def binomial_coeff(n, k):
    return factorial(n) / (factorial(k) * factorial(n - k))

def hyper_geometric_pmf(k, K, n, N):
    binom1 = binomial_coeff(K, k)
    binom2 = binomial_coeff(N-K, n - k)
    binom3 = binomial_coeff(N, n)
    return (binom1 * binom2) / binom3

### MAIN ---------------------------------------------------------------------------------------

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
    ## We will simply set P(N) = 1
    k10 = 10
    k15 = 15
    K = 100
    n = 60

    ## Plot posterior P(N|k) for k = 10 and k = 15 for a flat prior, as well as prior P(N) = 1/N
    ## Do estimator values for N differ between likelihood and posterior for flat hhv non-flat prior??

    fig, ax = plt.subplots()

    offset = 120
    N_vals = np.arange(offset,2000,1)


    likelihood_vals10 = stats.hypergeom.pmf(k10, N_vals, K, n)
    likelihood_vals15 = stats.hypergeom.pmf(k15, N_vals, K, n)

    ax.set(xlabel = 'N', ylabel = r'const $\times$ posterior')
    ax.plot(N_vals, likelihood_vals10, label = r'$P(N|k = 10)$, $P(N) = const$', color = 'teal')
    ax.plot(N_vals, likelihood_vals15, '--', label = r'$P(N|k = 15)$, $P(N) = const$', color = 'teal')

    ## Calculating posterior using prior P(N) = 1/N

    prior_vals = 1/N_vals
    norm_const = 0.002
    posterior10 = likelihood_vals10 * prior_vals / norm_const
    posterior15 = likelihood_vals15 * prior_vals / norm_const


    ax.plot(N_vals, posterior10, label = r'$P(N|k = 10)$, $P(N)=\frac{1}{N}$', color = 'navy')
    ax.plot(N_vals, posterior15, '--', label = r'$P(N|k = 15)$, $P(N)=\frac{1}{N}$', color = 'navy')

    ax.legend()

    ## Extract most likely values for each distribution
    max_likelihood_10 = np.max(likelihood_vals10)
    max_likelihood_15 = np.max(likelihood_vals15)
    max_posterior_10_non_flat = np.max(posterior10)
    max_posterior_15_non_flat = np.max(posterior15)
    Nflat_max10 = np.argwhere(likelihood_vals10 == max_likelihood_10).flatten() + offset
    Nflat_max15 = np.argwhere(likelihood_vals15 == max_likelihood_15).flatten() + offset
    Nnon_flat_max10 = np.argwhere(posterior10 == max_posterior_10_non_flat).flatten() + offset
    Nnon_flat_max15 = np.argwhere(posterior15 == max_posterior_15_non_flat).flatten() + offset

    print("Most likely posterior (=likelihood) values for flat prior for k = 10, k = 15: ",\
         Nflat_max10, Nflat_max15)
    print("Most likely posterior (!=likelihood) values for prior = 1/N for k = 10, k = 15: ",\
        Nnon_flat_max10, Nnon_flat_max15)

    print(f'Including the prior P(N) = 1/N naturally shifts the values of the posterior towards\
        smaller values of N')


    ## FISH EXERCISE
    # You want to est no. of fish in lake. We know that fish prefer 10 \pm 1 m^3 water alone, and that
    # V_lake = 5000 \pm 300 m^3. N_fish = V_lake / V_fish
    V_fish = 10
    dV_fish = 1.0
    V_fish2 = 9.2
    dV_fish2 = .2
    V_lake = 5000
    dV_lake = 300

    N_fish = V_lake / V_fish
    dN_fish_dV_lake = N_fish / V_lake
    dN_fish_dV_fish = - N_fish / V_fish
    dN_fish = np.sqrt(dN_fish_dV_lake ** 2 * dV_lake ** 2 + dN_fish_dV_fish ** 2 * dV_fish ** 2)

    print("Prior mean and  uncertainty on number of fish: ", N_fish, dN_fish)
    #Having a mean and an std, and assuming gaussian prior, we have an estimate for prior. P(N)
    fi2, ax2 = plt.subplots(figsize = (12,9))
    N_vals = np.arange(100,900)
    K = 50
    n = 30
    k =  np.arange(4,9)
   # norm_likelihood = 20
   # norm_posterior = 0.01

    prior = stats.norm.pdf(N_vals, loc = N_fish, scale = dN_fish)
    colors = ['teal', 'navy', 'plum', 'olivedrab', 'black']
    for i, successes in enumerate(k):
        likelihood_vals = stats.hypergeom.pmf(successes, N_vals, K, n)
        norm_likelihood = 1 / np.sum(likelihood_vals)
        likelihood_vals *=  norm_likelihood
        posterior_vals = prior * likelihood_vals 
        norm_posterior = 1 / np.sum(posterior_vals)
        posterior_vals *= norm_posterior
        # For K = 50, n = 30 and k = 4, answer

        ax2.plot(N_vals, prior, '.-', label = f'Prior for k = {successes}', color = colors[i])
        ax2.plot(N_vals, likelihood_vals, '--', label = f'Hypergeom. likelihood for k = {successes}', color = colors[i])
        ax2.plot(N_vals, posterior_vals, '-', label = f'Posterior for k = {successes}', color = colors[i])
        ax2.set(xlabel = 'Total number of fish', ylabel = 'Probability')
        ax2.legend(fontsize = 14)


    ## Repeat for k = 4,8 with uncertainty doubled, tripled
    fig3, ax3 = plt.subplots(ncols = 2, figsize = (16,3))
    ax3 = ax3.flatten()
    k = [4,8]
    dN = dN_fish * np.arange(1,4)


    for j, successes in enumerate(k):
        for i, std in enumerate(dN):
            prior = stats.norm.pdf(N_vals, loc = N_fish, scale = std)
            likelihood_vals = stats.hypergeom.pmf(successes, N_vals, K, n)
            norm_likelihood = 1 / np.sum(likelihood_vals)
            likelihood_vals *=  norm_likelihood
            posterior_vals = prior * likelihood_vals 
            norm_posterior = 1 / np.sum(posterior_vals)
            posterior_vals *= norm_posterior
            # For K = 50, n = 30 and k = 4, answer

   
            ax3[j].plot(N_vals, prior, 's', label = f'Prior for sigma_prior = {i+1}dN_fish', color = colors[i], markersize = 0.2)
            ax3[j].plot(N_vals, likelihood_vals, '--', label = f'Hypergeom. for sigma_prior = {i+1}dN_fish', color = colors[i])
            ax3[j].plot(N_vals, posterior_vals, '-', label = f'Posterior for sigma_prior = {i+1}dN_fish', color = colors[i])
            ax3[j].set(xlabel = 'Total number of fish', ylabel = 'Probability', title = f'For k = {successes}')
            ax3[j].legend(fontsize = 10)


    ## Changing V_fish --> 9.2 \pm 0.2 reduces the std of the prior, thus increasing the influence of prior influence and pulling the
    # posterior closer to the prior densities




    # How sensitive is posterior to form/values in prior?
    # Repeat for k = 4 to k = 8 
    # If Prior uncertainty is doubled/tripled, how much closer
    #..is likelihood value to posterior for k = 4-8
    # what if V_fish = 9.2 \pm 0.2




    plt.show()



if __name__ == '__main__':
    main()
