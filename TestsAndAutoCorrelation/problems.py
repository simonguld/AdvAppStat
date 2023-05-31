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

def P1():
    ## Hypothesis testing using poisson and binned likelihoods
    ## gen. 10^5 data pseudo experiments of monte carlo data for the cases
    # no signal and background with mean rate = 0.1, 10, 1000
    # background = 1000 and signal in first bin = 100, 200
    # gen. H0 = no signal / H1 = signal in first bin and plot test statistic
    # iff time calc p-val

    Nbins = 100
    Npoints = 100_000

    bck_rate = [0.1, 10, 1000]
    sig_rate = [100, 1000]

    def test_statistic(events_in_first_bin, Npoints, Nbins):
        if events_in_first_bin == 0:
            val = 2 * (Npoints - events_in_first_bin) \
            * np.log((Nbins * (Npoints - events_in_first_bin))/(Npoints * (Nbins - 1)))
        elif Npoints - events_in_first_bin == 0: 
            val = 2 * events_in_first_bin * np.log(Nbins / Npoints * events_in_first_bin) 
        else:
            val = 2 * events_in_first_bin * np.log(Nbins / Npoints * events_in_first_bin) \
            + 2 * (Npoints - events_in_first_bin) \
            * np.log((Nbins * (Npoints - events_in_first_bin))/(Npoints * (Nbins - 1)))
        return val
    test_statistic_vectorized = lambda x1: test_statistic(x1, Npoints, Nbins)
    fig_bck, ax_bck = plt.subplots(ncols = 3)
    ax_bck = ax_bck.flatten()
    fig_sig, ax_sig = plt.subplots(ncols = 2)
    ax_sig = ax_sig.flatten()

    for i, ax in enumerate(ax_bck):
        mc_data = stats.poisson.rvs(mu = bck_rate[i], size = [Npoints, Nbins])
        range = (np.min(mc_data), np.max(mc_data))
        count, edges, _ = ax.hist(mc_data[0], range = range, bins = Nbins, histtype = 'stepfilled',\
                                  ec = 'black', alpha = .3, label = f'mu_bck = {bck_rate[i]}')
        ax.legend()

    if 0:
        for i, ax in enumerate(ax_bck):
            mc_data_bck = stats.poisson.rvs(mu = bck_rate, size = Npoints)

            range = (np.min(mc_data), np.max(mc_data))
            count, edges, _ = ax.hist(mc_data, range = range, bins = Nbins, histtype = 'stepfilled',\
                                    ec = 'black', alpha = .3, label = f'mu_bck = {bck_rate[i]}')
            ax.legend()



    plt.show()

def P2():
    ## gen MC data isotropically.
    # derivate two point autocorrelation for the dist.
    # what dist do you expect??


    projections = ['aitoff', 'hammer', 'lambert', 'mollweide']


    ## How to to stereographical projections:
    ## gen points (theta, phi) in [-]
    Npoints = 100
    #azimuth = 2 * np.pi * (np.random.rand(Npoints) - 0.5)
    azimuth = 2 * np.pi * (np.random.rand(Npoints))
    #polar = 1 * np.pi * ( np.random.rand(Npoints) - 0.5)
    polar = np.arccos( 2 * (np.random.rand(Npoints)) - 1) - np.pi/2
    polar = np.arccos( 2 * (np.random.rand(Npoints)) - 1)
    points = np.r_['0,2', azimuth, polar].T
    points_dipol = np.r_['0,2', np.linspace(0,2*np.pi,Npoints), np.linspace(0,np.pi,Npoints)].T
 
    def spherical_dot_prod(azi1, pol1, azi2, pol2):
        val = np.sin(pol1) * np.sin(pol2) * np.cos(azi2 - azi1) + np.cos(pol1) * np.cos(pol2)
        return val
    
    def cum_auto_corr(angles, cutoff):
        """angle has format [Npoints, (azimuth, polar)]
        """
        Npoints, _ = angles.shape
        val = np.empty([Npoints, Npoints])
        for i, ang in enumerate(angles):
            val[i,:] = np.sin(ang[1]) * np.sin(angles[:,1]) * np.cos(ang[0] -angles[:,0]) + np.cos(ang[1]) * np.cos(angles[:,1]) - cutoff
            val[i,i] = -1

   
        mask = (val > 0)
        sum = len(val[mask])
        multiplier = 1 / (Npoints * (Npoints - 1))
        return multiplier * sum

    if 0:
        cutoff_arr = np.linspace(-1, 1, 10)
        cum_vals = np.empty_like(cutoff_arr)
        cum_vals_polar = np.empty_like(cum_vals)
        for i, cutoff in enumerate(cutoff_arr):
            cum_vals[i] = cum_auto_corr(points, cutoff)
            cum_vals_polar[i] = cum_auto_corr(points_dipol,cutoff)

        fig0, ax0 = plt.subplots()
        ax0.plot(cutoff_arr, cum_vals, label = 'Isotropic')
        ax0.plot(cutoff_arr, cum_vals_polar, label = 'Dipole')

        print(stats.ks_2samp(cum_vals, cum_vals_polar))

    if 0:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = 'aitoff')
        plt.grid(True)
        plt.scatter(azimuth, polar, marker = 'o', s = 3, alpha = .3)
    
    
    Ntrials = 100
    Npoints = 100
    Nresolution = 100
    cutoff_arr = np.linspace(-1, 1, Nresolution)
    KS_vals = []
   # perfect_distribution = np.linspace(1,0, Nresolution)

    Nperfect = 10_000
    perfect_distribution = np.empty(Nperfect)
    cutoff_perfect = np.linspace(-1,1, Nperfect)
    azimuth = 2 * np.pi * (np.random.rand(Npoints))
    polar = np.arccos( 2 * (np.random.rand(Npoints)) - 1)
    points = np.r_['0,2', azimuth, polar].T
    for i, cutoff in enumerate(cutoff_perfect):
        perfect_distribution[i] = cum_auto_corr(points, cutoff)


    fig2,ax2=plt.subplots()
    ax2.plot(cutoff_perfect, perfect_distribution)

    for n in np.arange(Ntrials):
        
        azimuth = 2 * np.pi * (np.random.rand(Npoints))
        polar = np.arccos( 2 * (np.random.rand(Npoints)) - 1)
        points = np.r_['0,2', azimuth, polar].T

        
        cum_vals = np.empty_like(cutoff_arr)
        for i, cutoff in enumerate(cutoff_arr):
            cum_vals[i] = cum_auto_corr(points, cutoff)
        ax2.plot(cutoff_arr, cum_vals, label = 'Isotropic')
        KS_vals.append(stats.ks_2samp(cum_vals, perfect_distribution)[0])

    print(KS_vals)
    fig1, ax1 = plt.subplots()
    bins = int(Ntrials / 5)
    range = (np.min(KS_vals), np.max(KS_vals))
    ax1.hist(KS_vals, histtype='stepfilled', alpha=.4, bins = bins, lw = 1.5)

    ax1.legend()
    fig1.tight_layout()

    if 0:

        fig = plt.figure()
        ax = fig.add_subplot(111, projection = 'mollweide')
        plt.grid(True, **dict (lw = 3))
        ax.scatter(azimuth, polar, marker = 'o', s = .3, alpha = .3)
   

    plt.show()


def P3():
    pass

def P4():
    pass

def main():
     ## Set which problems to run
    p1, p2, p3, p4 = False, True, True, False
    problem_numbers = [p1, p2, p3, p4]
    f_list = [P1, P2, P3, P4]

    for i, f in enumerate(f_list):
        if problem_numbers[i]:
            print(f'\nPROBLEM {i + 1}:')
            f()



if __name__ == '__main__':
    main()
