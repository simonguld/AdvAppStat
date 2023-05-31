# Author: Simon Guldager Andersen
# Date (latest update): 

### SETUP ------------------------------------------------------------------------------------

## Imports:
import os, sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats, integrate, optimize
from iminuit import Minuit
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cycler import cycler

sys.path.append('Appstat2022\\External_Functions')
from ExternalFunctions import Chi2Regression, BinnedLH, UnbinnedLH
from ExternalFunctions import nice_string_output, add_text_to_ax    # Useful functions to print fit results on figure


## Change directory to current one
os.chdir('AdvAppStat\Likelihood')


### FUNCTIONS ----------------------------------------------------------------------------------

def gaussian_LH(x, mean, std):
    return  1 / (np.sqrt(2 * np.pi) * std) * np.exp(-0.5 * (x-mean) ** 2 / std ** 2)


def rejection_sampling_uniform(function, fmax, bounds, Npoints, verbose = True):

    ## ALGORITHM: The goal is to x1, ...., xn points distributed according to f on domain D
    # 1) generate a point x in D distributed according to a probability distribution g/area(g) (so g can be any curve...) enclosing f, ie f <= g on D
    # 2) generate uniformly a random point u in [0,g(x)]
    # 3) if u<f(x), keep x. 
    # 4) Rinse and repeat until desired number of points has been aquired

    # generate values according to f using rejection samping
    r = np.random

    xmin, xmax = bounds[0], bounds[1] 

    ## Using rejection/accepting method with both a constant pdf as well as 1/(1+x)
    x_accepted = np.empty(0)
    N_try = int(3 * Npoints)
    N_accum = 0
  

    while x_accepted.size < Npoints:
        # Construct N_points points by accepting/rejecting using a uniform pdf
        ## First, we construct N_try points uniformly on [xmin,xmax]
        r_vals = xmin + (xmax - xmin) * r.rand(N_try)
        ## Next, we construct another set of uniform random values in [0,fmax = y]
        #u_vals = fmax * r.rand(N_try)
        u_vals = r.uniform(0, fmax, size = N_try)
        ## Finally, we keep only the r_vals values satisfying u_vals < f(r_vals)
        mask = (u_vals < function(r_vals))
        vals = function(r_vals)

        x_accepted = np.r_['0', x_accepted, r_vals[mask]]

        # store total number of calculated samples
        N_accum += N_try
        # update N_try
        N_try = int(3 * (Npoints - x_accepted.size))

        if x_accepted.size > Npoints:
            x_accepted = x_accepted[:Npoints]

    eff_uni = Npoints / N_accum
    eff_err = np.sqrt(eff_uni*(1-eff_uni) / N_accum)
 
    if verbose:
        print("efficiency uniform: ", f'{eff_uni:6.3f}', "\u00B1 ", f'{eff_err:6.3f}')

    return x_accepted, eff_uni, eff_err


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
    ## Ex0.5: Gen. 4 points from 2 Gaussian distributions as below
    loc1, scale1 = 1.25, np.sqrt(0.11)
    loc2, scale2 = 1.30, np.sqrt(0.50)
    samp1 = stats.norm.rvs(loc = loc1, scale = scale1, size = 4)
    samp2 = stats.norm.rvs(loc = loc2, scale = scale2, size = 4)


    # Plot the points
    fig, ax = plt.subplots()

    gauss1 = lambda x: stats.norm.pdf(x, loc = loc1, scale = scale1)
    gauss2 = lambda x: stats.norm.pdf(x, loc = loc2, scale = scale2)

    ax.plot(samp1, gauss1(samp1), 'x',label = r'$\mu = 1.25$, $\sigma = \sqrt{0.11}$')
    ax.plot(samp2, gauss2(samp2), 's',label = r'$\mu = 1.30$, $\sigma = \sqrt{0.50}$')

    ax.legend()
    fig.tight_layout()

    ## Given the four points below, does Gaussian1 or Gaussian2 maximize likelihood??
    x = np.array([1.01, 1.30, 1.35, 1.44])

    likelihood1 = 1
    likelihood2 = 1

    for point in x:
        likelihood1 *= gauss1(point)
        likelihood2 *= gauss2(point)

    print('Likelihood for Gauss 1: ', likelihood1)
    print('Likelihood for Gauss 2: ', likelihood2)


    ##Exc 1: Simulate 50 values dist. according to gaussian with mu = 0.2, sigma = 0.1
    # Calc logL using minimizer
    # Calc logL by doing a raster scan
    # Compare these values to likelihood of true input values for multiple iterations. Are the better or worse? WHY?
    mu, sig = .2, .1

    sample = stats.norm.rvs(loc = mu, scale = sig, size = 500)
    gauss = lambda x: stats.norm.pdf(x, loc = mu, scale = sig)


    logL_true = 1
    logL_minuit = 1

    logL_true = np.sum(np.log(gauss(sample)))

    UnbinnedLH_object = UnbinnedLH(gaussian_LH, sample, bound = (mu - 4 * sig, mu + 4 * sig), extended = False)
    fit =  Minuit(UnbinnedLH_object, mean = mu, std = sig)
    fit.errordef = Minuit.LIKELIHOOD
    fit.migrad()
    mu_fit, sig_fit = fit.values['mean'], fit.values['std']
    logL_minuit = fit.fval 


    ## Perform Raster scan, i.e. calc logL for all values in product range of mu and sig, and plot landscape
    width_sig = 0.04
    width_mu = 0.04
    Npoints = 100

    mu_vals = np.linspace(mu - width_mu, mu + width_mu, Npoints)
    sig_vals = np.linspace(sig - width_sig, sig + width_sig, Npoints)

    MU, SIG = np.meshgrid(mu_vals, sig_vals)
    gauss_vec = lambda val: stats.norm.pdf(val, loc = MU, scale = SIG)

    logL_numeric = np.zeros([Npoints, Npoints], dtype = 'float')

    for val in sample:
        logL_numeric += np.log(gauss_vec(val))
   
    logL_numeric_max = np.max(logL_numeric)

    index_opt_numeric = np.unravel_index(np.argmax(logL_numeric, axis = None), logL_numeric.shape)
    logL_numeric_max = logL_numeric[index_opt_numeric]
    mu_opt_numeric, sig_opt_numeric = mu_vals[index_opt_numeric[1]], sig_vals[index_opt_numeric[0]]

    fig2, ax2 = plt.subplots()

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax2.imshow(logL_numeric, extent = (mu_vals[0], mu_vals[-1], sig_vals[-1], sig_vals[0]), cmap = 'viridis', vmin = logL_numeric_max/2)
    #ax2.plot()
    ax2.plot(mu_opt_numeric, sig_opt_numeric, 'kx', lw = 2, markersize = 12)
    fig2.colorbar(im, cax = cax, orientation = 'vertical')

    print("logL using actual parameters: ", logL_true)
    print("mu, std, logL using minuit: ", mu_fit, sig_fit, - fit.fval)
    print("mu, std, logL for Rasper search: ", mu_opt_numeric, sig_opt_numeric, logL_numeric_max)




    ## EXC 2: Given function f(x,alpha,beta) = 1 + alphax + beta x ^2, for alpha = beta = 0.5, gen. 2000 MC points for this function transformed into
    # a pdf over the range [-1,1].
    # Write your own likelihood function to fit the estimators alpha and beta using MC sample and num. minimizer on LLH or -LLH to get values and possibly errors

    Npoints = 1000
    Npoints2 = Npoints
    bounds = [-1,1]
    alpha_true, beta_true = 0.5, 0.5

    def func(x, alpha, beta):
        return np.maximum(1e-10, 1 + alpha * x + beta * np.power(x, 2))

    ##Normalize 
    norm_const = integrate.quad(np.vectorize(lambda x: func(x, alpha_true, beta_true)), bounds[0], bounds[1])[0]
    
    func_norm = lambda x, alpha, beta: 1/norm_const * func(x, alpha, beta)
    func_norm_ext = lambda x, alpha, beta, N: N * func_norm(x, alpha, beta)

    fmax = func_norm(bounds[1], alpha_true, beta_true)

    sample_values = rejection_sampling_uniform(lambda x: func_norm(x, alpha_true, beta_true), fmax, bounds, Npoints)[0]

    ## Check that MC values are reasonable
    fig3, ax3 = plt.subplots()
    bins = 100
    binwidth = (bounds[1] - bounds[0]) / bins
    x_vals = np.linspace(bounds[0], bounds[1], 1000)
    f_vals = Npoints * binwidth * func_norm(x_vals, alpha_true, beta_true)

    ax3.hist(sample_values, range = bounds, bins = 100, histtype = 'stepfilled')
    ax3.plot(x_vals, f_vals, '-', label = 'True params')

    ## Minimize with minuit for reference
    LH_unbinned_object = UnbinnedLH(func_norm_ext, sample_values, bound = bounds, extended = True)
    fit = Minuit(LH_unbinned_object, alpha = alpha_true, beta = beta_true, N = Npoints)
    fit.errordef = Minuit.LIKELIHOOD
    print(fit.migrad())
    alpha_fit, beta_fit, N_fit = fit.values['alpha'], fit.values['beta'], fit.values['N']
    print("Minuit alpha beta -LLH: ", fit.values['alpha'], fit.values['beta'], fit.fval)

    ## Apparently, the Minuit LLH is calc as the max logLL - Npoints
    print("- (LogLH - Npoints (fitted parameters)): ", \
        -(np.sum(np.log(func_norm_ext(sample_values, alpha_fit, beta_fit, N_fit))) - Npoints))
    print("-(LogLH - Npoints (True parameters)): ", \
        -np.sum(np.log(func_norm_ext(sample_values, alpha_true, beta_true, Npoints))) + Npoints)
    
    ## Use minuit this time unextended
    

    ## Def -log likelihood function

    def func(x, alpha, beta):
        return np.maximum(0, 1 + alpha * x + beta * np.power(x, 2))

    func_norm = lambda x, args: 1 / (2 * (1 + args[1] / 3)) * func(x, args[0], args[1])
    logL_func= lambda args: - np.sum(np.log(func_norm(sample_values,  args)))

    print("\n Unextended -logLL for true params: ", logL_func([alpha_true, beta_true]))


    res = optimize.minimize(logL_func, np.array([0.5,.5]), bounds = [(0.2,0.8),(0.2,0.8)])
    print("Scipy alpha beta -LL: ", res.x, "  ", logL_func(res.x))


    ## DO raster scan
     ## Perform Raster scan, i.e. calc logL for all values in product range of mu and sig, and plot landscape
    width_alpha = 0.3
    width_beta = 0.3
    Npoints = 100

    alpha_vals = np.linspace(alpha_true - width_alpha, alpha_true + width_alpha, Npoints)
    beta_vals = np.linspace(beta_true - width_beta, beta_true + width_beta, Npoints)

    ALPHA, BETA = np.meshgrid(alpha_vals, beta_vals)
    
    func_norm_vec = lambda x: 1 / (2 * (1 + BETA / 3))  * func(x, ALPHA, BETA)
    func_norm = lambda x, alpha, beta: 1 / (2 * (1 + beta / 3))  * func(x, alpha, beta)
    logL_numeric = np.zeros([Npoints, Npoints], dtype = 'float')


    LH_unbinned_object = UnbinnedLH(func_norm, sample_values, bound = bounds, extended = False)
    fit = Minuit(LH_unbinned_object, alpha = alpha_true, beta = beta_true, verbose = 5)
    fit.errordef = Minuit.LIKELIHOOD
    fit.migrad()
    print("Unexteded alpha beta -LL Minuit: ", fit.values[:], fit.fval)


    for val in sample_values:
        logL_numeric += np.log(func_norm_vec(val))
   
    logL_numeric_max = np.max(logL_numeric)

    index_opt_numeric = np.unravel_index(np.argmax(logL_numeric, axis = None), logL_numeric.shape)
    logL_numeric_max = logL_numeric[index_opt_numeric]
    alpha_opt_numeric, beta_opt_numeric = alpha_vals[index_opt_numeric[1]], beta_vals[index_opt_numeric[0]]
    print("raster alpha beta: ", alpha_opt_numeric, beta_opt_numeric)
    fig4, ax4 = plt.subplots()

    divider = make_axes_locatable(ax4)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax4.imshow(logL_numeric, extent = (alpha_vals[0], alpha_vals[-1], beta_vals[-1], beta_vals[0]), \
        cmap = 'viridis') #, vmin = logL_numeric_max/2)
    #ax2.plot()
    ax4.plot(alpha_opt_numeric, beta_opt_numeric, 'kx', lw = 2, markersize = 12)
    fig3.colorbar(im, cax = cax, orientation = 'vertical')

    print('Raster scan alpha beta -LL: ', alpha_opt_numeric, beta_opt_numeric, -logL_numeric_max)

    ## Lav lnL funktion og minimer
    ## Lav Raster scan og verify
    ## Plot the path of your minimizer as it 'steps' through the landscape




    plt.show()

if __name__ == '__main__':
    main()
