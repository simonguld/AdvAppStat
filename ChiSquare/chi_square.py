# Author: Simon Guldager Andersen
# Date (latest update): 

### SETUP ------------------------------------------------------------------------------------

## Imports:
import os, sys
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib import rcParams
from cycler import cycler

## Change directory to current one
os.chdir('AdvAppStat\ChiSquare')


### FUNCTIONS ----------------------------------------------------------------------------------

def calc_mean_std_sem(x, ddof = 1):
    """ returns mean, std, sem (standard error on mean)
    """
    Npoints = len(x)
    mean = x.mean()
    std = x.std(ddof = ddof)
    sem = std / np.sqrt(Npoints)
    return mean, std, sem

def calc_chi2(y, dy, exp):
    chi2 = np.sum((y-exp) ** 2 / (dy) ** 2)
    Ndof = len(y)
    p = stats.chi2.sf(chi2,Ndof)
    return chi2, Ndof, p

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
rcParams['figure.figsize'] = (6,6)
rcParams['axes.prop_cycle'] = cycler(color = ['teal', 'navy', 'coral', 'plum', 'purple', 'olivedrab',\
         'black', 'red', 'cyan', 'yellow', 'khaki','lightblue'])
np.set_printoptions(precision = 5, suppress=1e-10)

def main():


    if 1:
    # data = np.genfromtxt('FranksNumbers.txt', dtype = 'str', skip_header = 1, invalid_raise = False, missing_values = ' ', usemask = False, filling_values='0.0')
        data = pd.read_csv('FranksNumbers.txt', skip_blank_lines = True, dtype = 'str')
        

        Ndatasets = 5
        Nvariables = 2

        # The data contains 5 data sets. In the following, we separate them
        data1, data2, data3, data4, data5 = [],[],[],[],[]
        data_list = [data1, data2, data3, data4, data5]

        n = -1
        for i, row in enumerate(data.values):
            if row == f'Data set {n+2}':
                n +=1
                continue

            row = str(row)[2:-2]
            x_val, y_val = row.split(sep="\\t")[0], row.split(sep="\\t")[1]
   
            # Note that the format of each data set is [x1, y1, x2, y2, ...]
            data_list[n].append(float(x_val))
            data_list[n].append(float(y_val))

            ## Calc mean and var for each data set
            ## using exp = 0.48x + 3.02, calc x^2 with
            # 1) dx = 1.22  2) dx = sqrt(N)

        for i in range(Ndatasets):
            data = np.array([data_list[i]]).flatten()
            Npoints = len(data[:])
            # Extract x and y values
            x, y = data[np.arange(0, Npoints, 2)], data[np.arange(1, Npoints, 2)]

            # Calc mean and var for x and y
            mean_x, std_x, sem_x = calc_mean_std_sem(x)
            mean_y, std_y, sem_y = calc_mean_std_sem(y)

            # Calc chi2 using dx = sqrt(Npoints) and dx = 1.22 and exp = 0.48x+3.02
            exp = 0.48 * x + 3.02
            err_const = 1.22

            poiss_dx = np.sqrt(Npoints / 2)
            poiss_chi2, poiss_Ndof, poiss_p = calc_chi2(y, poiss_dx, exp)
            const_chi2, const_Ndof, const_p = calc_chi2(y, err_const, exp)

            print(f'\nFOR DATASET {i+1}: \n')
            print("Mean, SEM and var of x: ", np.round(mean_x,4), "\u00B1", np.round(sem_x,4), "  ", np.round(std_x ** 2, 4))
            print("Mean, SEM and var of y: ", np.round(mean_y,4), "\u00B1", np.round(sem_y,4), "  ", np.round(std_y ** 2, 4))
            print("\n")
            print("chi2, Ndof and p-val with Poisson errors: ", poiss_chi2, poiss_Ndof, poiss_p)
            print(f"chi2, Ndof and p-val with const err = {err_const}: ", const_chi2, const_Ndof, const_p)

            if poiss_p < const_p:
                best = 'Poisson errors'
            else:
                best = f'Const errors = {err_const}'

            print(f"We see that using {best} results in best agreement with data")



if __name__ == '__main__':
    main()
