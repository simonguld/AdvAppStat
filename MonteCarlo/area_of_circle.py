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

## Change directory to current one
os.chdir('...')


### FUNCTIONS ----------------------------------------------------------------------------------

def simulate_circle_area(radius, Npoints, visualize = False, ax = None):
        rvs = np.random
        rvs.seed()
        dimensions = 2
        
        radius = 5.2
        square_vals = radius * (rvs.rand(dimensions, Npoints) - 0.5) * 2

        mask = (square_vals[0] ** 2 + square_vals[1] ** 2 < radius ** 2)

        Npoints_in_circle = len(np.argwhere(mask == True).flatten())
        frac_within_circle = Npoints_in_circle / Npoints
        err_frac_within_circle = np.sqrt(frac_within_circle * (1 - frac_within_circle) / Npoints)

        area_square = (2 * radius) ** 2
        area_circle_sim = frac_within_circle * area_square
        err_area_circle_sim = err_frac_within_circle * area_square

        if visualize:
            ax.set(xlabel = 'x', ylabel = 'y', title = f'Uniform values simulated in the ranges\
                 {[-radius,radius]}x{[-radius,radius]}')
 
            circle_vals_x = square_vals[0][mask]
            circle_vals_y = square_vals[1][mask]
            x_vals_without = square_vals[0][~mask]
            y_vals_without = square_vals[1][~mask]


            ax.add_artist(plt.Circle((0,0), radius, fill = False))
            ax.add_artist(plt.Rectangle((0,0),2 * radius, 2 * radius, fill = False, lw = 2))
        # ax.add_artist(plt.box)

            ax.scatter(circle_vals_x, circle_vals_y, alpha = 0.5)
            ax.scatter(x_vals_without, y_vals_without, alpha = 0.5)


        return area_circle_sim, err_area_circle_sim


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
    ## Assume that the value of pi is unkown.
    ## 1) Estimate the area of the cirlce using only x**2+y**2=r**2 and a random number generator
    ## 2) Est area of circle with r = 5.2 and compare with true value
    ## Visualize your method
    ## (optional) Estimate the area using a MC-sampler in only 1 dimension



  
    Npoints = 10000
    radius = 5.2

    fig,ax = plt.subplots()
    area_circle_sim, err_area_circle_sim = simulate_circle_area(radius, Npoints, visualize = True, ax = ax)
    area_circle_true = np.pi * radius ** 2

    print(f'Expected area for circle with radius = {radius}: ', area_circle_true)
    print(f'Simulated value of area using {Npoints} points: ', area_circle_sim, "\u00B1", err_area_circle_sim)
    print('Relative error = ', np.abs(area_circle_sim - area_circle_true) / area_circle_true)

    fig.tight_layout()



    ## Using 100 samples, calc. area 1000 times and plot results. Is CLT satisfied? Are there gaps? Should there be?
    # plot the same 1000 values using binwidth = 3, 1, 0.1

    Npoints = 100
    Ntrials = 1000
    area_list = np.empty(Ntrials)

    for i in np.arange(Ntrials):
        area_list[i] = simulate_circle_area(radius, Npoints)[0]
    
    fig2, ax2 = plt.subplots()
    shift = 15
    range = (np.pi * radius ** 2 - shift-3, np.pi * radius ** 2 + shift)
    binwidth = np.array([3, 1, 0.1])
    bins = ((range[1] - range[0]) / binwidth).astype('int')

    for i, bin in enumerate(bins):
        ax2.hist(area_list, range = range, bins = bin, histtype = 'step', alpha = .7, label = f'Binwidth = {binwidth[i]}')
        

    ## CONCL:: Trouble is, choosing the batch size = 100, we put a cap on the maximal precision that we can attain.
    ## SOLUTION:: Increase batch size.
    
    ax2.legend()

    ## EX3: Assuming that A = pi r^2, estimate pi using 10, 100, ..., 1e6 simulations and plot

    Nsimulations = np.logspace(3,5, num = 100).astype('int')

    pi_list = np.empty(len(Nsimulations), dtype = 'float')
    pi_err_list = np.empty_like(pi_list, dtype = 'float')

    for i,N in enumerate(Nsimulations):
        area, err_area = simulate_circle_area(radius, N)
        pi_list[i] = area / radius ** 2
        pi_err_list[i] = err_area / radius ** 2

    fig3, ax3 = plt.subplots()
    ax3.errorbar(np.log10(Nsimulations), pi_list, pi_err_list, fmt = 'k.', capsize = 1, capthick = 1, lw = 1)
    ax3.plot([np.min(np.log10(Nsimulations)), np.max(np.log10(Nsimulations))], [np.pi, np.pi], '-', label = 'True value')
    ax3.set(xlabel = r'$\log_{10}(N_{simulations})$', ylabel = r'Estimate of $\pi$', ylim = [3.04,3.24])


    plt.show()

if __name__ == '__main__':
    main()
