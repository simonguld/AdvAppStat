# Author: Simon Guldager Andersen
# Date (latest update): 

### SETUP ------------------------------------------------------------------------------------

## Imports:
import os, sys
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, integrate, interpolate, optimize
from iminuit import Minuit
from matplotlib import rcParams
from cycler import cycler

sys.path.append('Appstat2022\\External_Functions')
from ExternalFunctions import Chi2Regression, BinnedLH, UnbinnedLH
from ExternalFunctions import nice_string_output, add_text_to_ax    # Useful functions to print fit results on figure


## Change directory to current one
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)



### FUNCTIONS ----------------------------------------------------------------------------------



### MAIN ---------------------------------------------------------------------------------------

def P1():
    """
    Use a fixed length hyper cube KDE (constant in |x-h|) in 1D with bandlength h=1.5 
    Calculate P(x) for x = (6, 10.1, 20.499, 20.501)
    """
    data = np.array([1, 2, 5, 6, 12, 15, 16, 16, 22, 22, 22, 23], dtype='float')
    eval_points = np.array([6, 10.1, 20.499, 20.501])

    bandwidth = 1.5

    def hypercube_kde(x, x0, bandwidth):
        """
        x can be a vector, x0 is a scalar
        """
        mask = np.abs(x - x0) < bandwidth
        val = np.zeros_like(x, dtype = 'float')
        val[mask] = 1 / (2 * bandwidth)
        return val
    
    
    def hypercube_pde(x_data, bandwidth):
        Npoints = len(x_data)
        def pdf(x):
            # vectorize KDE 
            kde_vectorized = np.vectorize(lambda x, x0: hypercube_kde(x, x0, bandwidth))

            ## initialize pdf 
            val = np.zeros_like(x, dtype = 'float')
            for data_point in x_data:
                val += kde_vectorized(x, data_point)
            # Normalize pdf
            val = val / Npoints
            return val
        return pdf

    def epanechnikov_kde(x, x0, bandwidth):
        """
        x can be a vector, x0 is a scalar
        """
        u =  np.abs(x - x0) / bandwidth
        mask = (u < 1)
        val = np.zeros_like(x, dtype = 'float')
        val[mask] = 3 / (4 * bandwidth) * ( 1 - u[mask] ** 2)
        return val
    
    
    def epanechnikov_pde(x_data, bandwidth):
        Npoints = len(x_data)
        def pdf(x):
            # vectorize KDE 
            kde_vectorized = np.vectorize(lambda x, x0: epanechnikov_kde(x, x0, bandwidth))

            ## initialize pdf 
            val = np.zeros_like(x, dtype = 'float')
            for data_point in x_data:
                val += kde_vectorized(x, data_point)
            # Normalize pdf
            val = val / Npoints
            return val
        return pdf

    def gaussian_kde(x, x0, std):
        """
        x can be a vector, x0 is a scalar
        """
        val = 1 / (np.sqrt(2 * np.pi) * std) * np.exp(- 0.5 * (x - x0) ** 2 / std ** 2)
        return val
    
    def gaussian_pde(x_data, std):
        Npoints = len(x_data)
        def pdf(x):
        #    # vectorize KDE 
            kde_vectorized = np.vectorize(lambda x, x0: gaussian_kde(x, x0, std))

            ## initialize pdf 
            val = np.zeros_like(x, dtype = 'float')
            for data_point in x_data:
                val += kde_vectorized(x, data_point)  #kde_vectorized(x, data_point)
            # Normalize pdf
            val = val / Npoints
            return val
        return pdf


    ## construct pdf
    pdf = hypercube_pde(data,bandwidth)
  
    ## evaluate it for the given points
    for point in eval_points:
        print(f' hyperbox pdf(x = {point}) = ', pdf(point))


    ## EXC2: Redo but now Gaussian kernel with std = 3
    # Do Gaussian kernel by 1) writing yourself 2) using package
    # plot kde from -10,35 and plot individual kernel contributions too
    std = 3
    pdf_gauss = gaussian_pde(data, std)
    print(pdf_gauss(np.array([2,1,3,0,4])))

    fig,ax = plt.subplots()

    range= (-10, 35)
    binwidth = 1
    bins = int((range[1] - range[0]) / binwidth)
    x_vals = np.linspace(range[0], range[1], 1000)

    ax.hist(data, range = range, bins = bins, histtype='step', alpha=.6, density=True)
    ax.plot(x_vals, pdf_gauss(x_vals), label = 'PDF from gaussian KDEs')
    ax.plot(x_vals, pdf(x_vals), label = 'PDF from cube KDEs')

    for data_val in data:
        ax.plot(x_vals, gaussian_kde(x_vals, data_val, std), 'r--', lw = 1, alpha = .6)


    ## use package for Gauss KDE

    ## To use a preset bandwidth = std_gauss / data.std, do
    scipy_kde = stats.gaussian_kde(data, bw_method = std / data.std(ddof = 1))
    ## else use scott or silverman
    scipy_kde = stats.gaussian_kde(data, bw_method = 'silverman')
 
    print("k est: ", std / data.std(ddof = 1))
    print("k scipy silverman: ", scipy_kde.covariance_factor())
    fvals = scipy_kde.pdf(x_vals)
    ax.plot(x_vals,fvals, label = f'Scipy PDF bandwidth = {scipy_kde.covariance_factor():.2f}')


    ##EXC: Repeat but now with Epanechnikov kernel with bandwidth of your choosing
    bandwidth = 6

    print(epanechnikov_kde(0,1,1))
    
    pdf_epanechnikov = epanechnikov_pde(data, bandwidth)
    print(np.trapz(pdf_epanechnikov(x_vals), x_vals))
    ax.plot(x_vals, pdf_epanechnikov(x_vals), label = 'PDF from Epanechnikov KDEs')

    ax.legend()
    plt.show()


def P3():
    pass

def P4():
    pass

def P5():
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
    p1, p5 = True, True
    p_list = [p1, p5]
    f_list = [P1, P5]

    for i, problem in enumerate(f_list):
        print(f"\nPROBLEM {i+1}:")
        problem()





if __name__ == '__main__':
    main()
