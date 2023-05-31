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
sns.set_style("whitegrid")
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

    projections = ['aitoff', 'hammer', 'lambert', 'mollweide']
    if 0:
        for i in range(len(projections)):
            plt.figure()
            plt.subplot(111, projection = projections[i])
            plt.title(f'{projections[i]}')


    ## How to to stereographical projections:
    ## gen points (theta, phi) in [-]
    Npoints = 100000
    azimuth = 2 * np.pi * (np.random.rand(Npoints) - 0.5)
    #polar = 1 * np.pi * ( np.random.rand(Npoints) - 0.5)
    polar = np.arccos( 2 * (np.random.rand(Npoints)) - 1) - np.pi/2

    
    fig = plt.figure()
   # ax = fig.add_subplot(111, projection = 'aitoff')
    plt.grid(True)
    plt.scatter(azimuth, polar, marker = 'o', s = .3, alpha = .3)
    


    fig = plt.figure()
    ax = fig.add_subplot(111, projection = 'mollweide')
    plt.grid(True, **dict (lw = 3))
    ax.scatter(azimuth, polar, marker = 'o', s = .3, alpha = .3)
   
    fig = plt.figure()
   
    plt.show()


if __name__ == '__main__':
    main()
