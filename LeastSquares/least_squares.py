# Author: Simon Guldager Andersen
# Date (latest update): 

### SETUP ------------------------------------------------------------------------------------

## Imports:
import os, sys
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
os.chdir('AdvAppStat\LeastSquares')


### FUNCTIONS ----------------------------------------------------------------------------------

def get_statistics_from_fit(fitting_object, Ndatapoints):

    Ndof = Ndatapoints - len(fitting_object.values[:])
    chi2 = fitting_object.fval
    prop = stats.chi2.sf(chi2, Ndof)
    return Ndof, chi2, prop


def construct_ls_coeff_matrix(x, degree):
    coeff_matrix = x[:, np.newaxis] ** np.arange(degree + 1)
    return coeff_matrix

def polynomail_ls_func(coefficients, x, y, dy, degree):
    coeff_matrix = construct_ls_coeff_matrix(x, degree = degree)

    ls_val = np.sum((y - coeff_matrix @ coefficients) ** 2 / (dy) ** 2)
    return ls_val

def polynomial_func(x, params):
    # degree = len(params) - 1
    x_vals = x ** np.arange(len(params))
    return np.sum(params * x_vals)

def nonzero_polynomial_func(x, params):
    # degree = len(params) - 1
    x_vals = x ** np.arange(len(params))
    return np.max(np.sum(params * x_vals), 0)
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
    ## Exc 0: For the data below, use least squares to fit diff. order polynomials and calc chi^2 val
    x_vals = np.arange(6).astype('float')
    y_vals = np.array([0.0, .8, 0.9, 0.1, -0.8, -1.0])
    dy_vals = 0.25 * np.ones_like(y_vals)


    ## APP 1: Construct LS and minimize using Minuit
    ## APP 2: Use Householder QT and LS routine

    # APP 1:
    fig1, ax1 = plt.subplots()
    #Plot points
    ax1.errorbar(x_vals, y_vals, dy_vals, fmt = 'k.', elinewidth=1, capsize = 1, capthick = 1,  label = 'Data points')
    d = {}

    degrees = np.arange(5)

    for i, degree in enumerate(degrees):
        # Construct fitting function
        polynomail_ls_func_res = lambda coeff: polynomail_ls_func(coeff, x_vals, y_vals, dy_vals, degree)

        # Construct fitting object simply using 1 as parameter guesses
        fit = Minuit(polynomail_ls_func_res, np.ones(degree + 1))
        fit.errordef = Minuit.LEAST_SQUARES
        fit.migrad()

        ## Calc values of ls polynomial
        x_range = np.linspace(np.min(x_vals), np.max(x_vals), 1000)
        fit_val_func = np.vectorize(lambda x: polynomial_func(x, fit.values[:]))
   
        fit_vals = fit_val_func(x_range)

        # Get statistics
        Ndof, chi2, prop = get_statistics_from_fit(fit, len(x_vals))

        ax1.plot(x_range, fit_vals, label = f'LS fit degree {degree}')
        d0 = {f'P(chi2) deg. {degree}': prop}
        d.update(d0)
        text = nice_string_output(d, extra_spacing=2, decimals=3)
        add_text_to_ax(0.05, 0.35, text, ax1, fontsize=14)
        
    ax1.legend()
    fig1.tight_layout()


    # EXC 2: #Gen 10,100,... Gaussian values and feat each with 2nd and 3rd order polynomial.
    # --> assume negative predictions from poly is 0
    # --> calc chi2 for each combination
    # --> plot the resultant fits
    # --> see what happens for higher orer polynomials as Nsamples increases [chi^2 or chi2/nof as func of Nsamples??]
    # Where do fits give good 'fit' to a gaussian? How does this agreement chance with degree or Nsamples?
    Ntrials1, Ntrials2 = 3, 6
    Ndegrees1, Ndegrees2 = 8,14
    Nsamples = np.logspace(Ntrials1, Ntrials2, Ntrials2 - Ntrials1 + 1).astype('int')
    degrees = np.arange(Ndegrees1,Ndegrees2,1)
    stat_matrix = np.empty([Ntrials2 - Ntrials1 + 1, 3 * (Ndegrees2 - Ndegrees1)])
    sample_mean = 0
    sample_std = 1
    shift = 3
    range = [sample_mean - shift * sample_std, sample_mean + shift * sample_std]
    bin_width = 0.1
    bins = int((range[1] - range[0]) / bin_width)

    fig2, ax2 = plt.subplots(nrows = 2, ncols = 2, figsize = (20,16))
    ax2 = ax2.flatten()

    for j, N in enumerate(Nsamples):
        sample_vals = stats.norm.rvs(loc = sample_mean, scale = sample_std, size = N)
        count, edges, _ = ax2[j].hist(sample_vals, range = range, bins = bins, histtype = 'step', lw = 2, alpha = .5, label = f'Simulated values')
        ax2[j].set(xlabel = 'Sim. values', ylabel = 'Count', title = f'Sim. values and pol. fits for N = {N}', xlim = range, ylim = (-0.1*np.max(count), (1+0.1)*np.max(count)))
        x_vals = 0.5 * (edges[1:] + edges[:-1])
        dy = np.sqrt(count)
        mask = (count > 0)
        ax2[j].errorbar(x_vals, count, dy, fmt = 'k.', elinewidth = 1, capsize = 1, capthick = 1, alpha = .8)
        d = {}

        for i, degree in enumerate(degrees):
                 # Construct fitting function
            def polynomail_ls_func_res(coeff):
                val = polynomail_ls_func(coeff, x_vals[mask] - sample_mean, count[mask], dy[mask], degree)
                return np.max(val, 0)

            # Construct fitting object simply using 1 as parameter guesses
            fit = Minuit(polynomail_ls_func_res, 0.05 * np.ones(degree + 1))
            fit.errordef = Minuit.LEAST_SQUARES
            fit.migrad()
            

            ## Calc values of ls polynomial
            x_range = np.linspace(range[0], range[1], 1000)
            fit_val_func = np.vectorize(lambda x: nonzero_polynomial_func(x, fit.values[:]))
    
            fit_vals = fit_val_func(x_range - sample_mean)

            # Get statistics
            Ndof, chi2, prop = get_statistics_from_fit(fit, len(x_vals[mask]))
            stat_matrix[j, 3*i:3*i+3] = np.array([Ndof, chi2, prop])
            print(j, degree, fit.fmin.is_valid)
            print(chi2/Ndof, prop)

            fit_vals_test = fit_val_func(x_vals[mask])
            chi2_test = np.sum((fit_vals_test - count[mask]) ** 2/dy[mask] ** 2)
            
   
            ax2[j].plot(x_range, fit_vals, label = f'LS fit degree {degree}')
            d0 = {f'P(chi2) deg. {degree}': prop}
            d.update(d0)
        text = nice_string_output(d, extra_spacing=2, decimals=3)
        add_text_to_ax(0.65, 0.28, text, ax2[j], fontsize=10)
        ax2[j].legend(fontsize = 10, loc = 'upper left')
    fig2.tight_layout()




    plt.show()

if __name__ == '__main__':
    main()
