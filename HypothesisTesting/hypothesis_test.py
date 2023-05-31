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
os.chdir('AdvAppStat\HypothesisTesting')


### FUNCTIONS ----------------------------------------------------------------------------------



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
    ## For each data set, calc the ln ratio ratio and calc the p-val for x in [-1,1]
    # Null hypothesis is f = 1 + rhox + omega x^2
    ## alt hypothesis is f = 1 + rhox + omega x^2 - gamma x^5

    data1 = np.loadtxt("LLH_Ratio_2_data.txt", usecols = 0)
    data2 = np.loadtxt("LLH_Ratio_2a_data.txt", usecols = 0)

    Npoints1 = len(data1)
    Npoints2 = len(data2)
    bins = 100
    range = [-1, 1]

    print("Dataset 2 and 2a contains ", Npoints1, Npoints2, "points, respectively")


    def func1(x, alpha, beta):
        return np.maximum(1e-10, 1 + alpha * x + beta * np.power(x, 2))

    def func2(x, alpha, beta, gamma):
        return np.maximum(1e-10, 1 + alpha * x + beta * np.power(x, 2)) - gamma * x ** 5


    norm_const = lambda beta: 2 + 2 * beta/3
    
    func1_norm = lambda x, alpha, beta: 1 / (norm_const(beta)) * func1(x, alpha, beta)
    func2_norm = lambda x, alpha, beta, gamma: 1 / (norm_const(beta)) * func2(x, alpha, beta, gamma)


    fig1, ax1 = plt.subplots(ncols = 2)
    ax1 = ax1.flatten()

    print(data1.sum(), data2.sum(), data1.std(), data2.std())

    ax1[0].hist(data1, range = range, bins = bins, histtype = 'stepfilled', alpha = .3, label = 'Data set 2')
    ax1[1].hist(data2, range = range, bins = bins, histtype = 'stepfilled', color = 'red', alpha = .3, label = 'Data set 2a')

   
    alpha_guess, beta_guess, gamma_guess = 0.5, 0.5, 0.1
    for i, data in enumerate([data1, data2]):

        LH_unbinned_object1 = UnbinnedLH(func1_norm, data, bound = range, extended = False)
        fit1 = Minuit(LH_unbinned_object1, alpha = alpha_guess, beta = beta_guess)
        fit1.errordef = Minuit.LIKELIHOOD
        fit1.migrad()
        print("Parabolic fit values: ", fit1.values[:])
        print("Parabolic -LLH: ", fit1.fval)

        LH_unbinned_object2 = UnbinnedLH(func2_norm, data, bound = range, extended = False)
        fit2 = Minuit(LH_unbinned_object2, alpha = alpha_guess, beta = beta_guess, gamma = gamma_guess)
        fit2.errordef = Minuit.LIKELIHOOD
        fit2.migrad()

        print("5th order fit values: ", fit2.values[:])
        print("5th order -LLH: ", fit2.fval)

        deltaLLH = 2 * (fit1.fval - fit2.fval)
        print("- 2 (LLH_parab - LLH_5th) = ", deltaLLH)
        print("Ndof, p: ", 1, stats.chi2.sf(deltaLLH, 1))


        ## Plot
        x_vals = np.linspace(range[0], range[1], 1000)
        bin_width = 2/bins
        f1_vals = Npoints1 * bin_width * func1_norm(x_vals, *fit1.values[:])
        f2_vals = Npoints2 * bin_width * func2_norm(x_vals, *fit2.values[:])
        ax1[i].plot(x_vals, f1_vals, label = 'Parabolic fit')
        ax1[i].plot(x_vals, f2_vals, label = '5th order pol fit')
        ax1[i].legend()


    plt.show()
  

if __name__ == '__main__':
    main()
