# Author: Simon Guldager Andersen
# Date (latest update): 

### SETUP ------------------------------------------------------------------------------------

## Imports:
import os, sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats, interpolate, integrate
from matplotlib import rcParams
from cycler import cycler

## Change directory to current one
os.chdir('AdvAppStat\Interpolation')


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
    ## LOAD DATA, plot along with linear and cubic splines
    data = np.loadtxt('DustLog_forClass.dat.txt')
    Npoints = data.shape[0]

    fig1, ax1 = plt.subplots()

    ## Construct linear splines
    f_lin = interpolate.interp1d(data[:,0], data[:,1], kind = 'linear')
    f_cubic = interpolate.interp1d(data[:,0], data[:,1], kind = 'cubic')

    # construct x-vals on which to evaluate the spline functions
    dx = data[0,1] - data[0,0]
 
    x_vals = np.linspace(np.min(data[:,0]), np.max(data[:,0]), 10 * Npoints)

   
    ax1.set(xlabel = 'Depth (meters)', ylabel = 'LogLogger value')
    ax1.plot(data[:,0], data[:,1], '^', label = 'data points', markersize = 1.5)
    ax1.plot(x_vals, f_lin(x_vals), '-', label = 'Linear splines', lw = .9)
    ax1.plot(x_vals, f_cubic(x_vals), '--', label = 'Cubic splines', lw = .9)


    ax1.legend()
    fig1.tight_layout()


    ##EXC 2:
    pdf_data = np.loadtxt('SplineCubic.txt')

    range = [1e-5, 1]
    Npoints = pdf_data.shape[0]
    x_vals = np.linspace(range[0], range[1], 10 * Npoints)
    dy = np.sqrt(pdf_data[:,1])

    ## Construct spline functions
    f_lin = interpolate.interp1d(pdf_data[:,0], pdf_data[:,1], kind = 'linear')
    f_quad = interpolate.UnivariateSpline(pdf_data[:,0], pdf_data[:,1], k = 2, w = 1/dy, s = Npoints + np.sqrt(2*Npoints))
    f_cubic = interpolate.UnivariateSpline(pdf_data[:,0], pdf_data[:,1], k = 3, w = 1/dy, s =  Npoints + 0.5 * np.sqrt(2*Npoints))

    ## Check for normalization:
    A_lin = integrate.quad(f_lin, range[0], range[1], limit = 100)[0]
    A_quad = integrate.quad(f_quad, range[0], range[1], limit = 100)[0] 
    A_cubic = integrate.quad(f_cubic, range[0], range[1], limit = 100)[0]

    print("Area under curve for lin. spline: ", A_lin)
    print("Area under curve for quad spline: ", A_quad)
    print("Area under curve for cubic spline: ", A_cubic)

    fig2, ax2 = plt.subplots()

    
    ax2.set(xlabel = 'x', ylabel = 'Density value')
   # ax2.plot(pdf_data[:,0], pdf_data[:,1], 'x', label = 'data points', markersize = 3.5)
    ax2.plot(x_vals, f_lin(x_vals)/A_lin, '-', label = 'Linear splines', lw = 1.3)
    ax2.plot(x_vals, f_quad(x_vals)/A_quad, '.', label = 'Quadratic splines', lw = 1.3, markersize = 1.5)
    ax2.plot(x_vals, f_cubic(x_vals)/A_cubic, '--', label = 'Cubic splines', lw = 1.3)

    ax2.legend()
    fig2.tight_layout()

    ##EXC 3: Neutrino oscillations. At very small energies, the oscillation goes into 'rapid' oscillatoin, where experimentally the prob. is best app. by 0.5
    # At what energy does the oscillations stop being sinosoidal? 
    # Can you make a cubic spline function (using smoothness parameter) that matches osc. prob when sampling okay, and then averages to 0.5 at lower E
    E_cutoff = 0.4863
    data_neu = np.loadtxt('SplineOsc1.txt')

    fig3, ax3 = plt.subplots()
    Npoints = data_neu.shape[0]
    range = (np.min(data_neu[:,0]), np.max(data_neu[:,0]))

    ax3.set(xlabel = 'Energy', ylabel = 'Oscillation Probability', title = 'Neutrino flavour change probability')

    ## Construct spline functions
    x_vals = np.linspace(range[0], range[1], 10 * Npoints)
    f_lin = interpolate.interp1d(data_neu[:,0], data_neu[:,1], kind = 'linear')
    f_cubic = interpolate.interp1d(data_neu[:,0], data_neu[:,1], kind = 'cubic')
    df_cubic = np.abs(np.diff(f_cubic(data_neu[:,0])))
    weights = 1 / np.r_['0', df_cubic, np.array([df_cubic[-1]])]
    mask = (data_neu[:,0] < E_cutoff)
   # weights = np.ones_like(data_neu[:,0])
    weights[mask] = 20 ## forces polynomial  to stay fixed in [0,1]

    f_cubic = interpolate.UnivariateSpline(data_neu[:,0], data_neu[:,1], w = weights ** 2 , k = 3,\
        s = Npoints +2* np.sqrt(Npoints)) 
    plotting_mask = (x_vals < E_cutoff)

    ax3.plot(data_neu[:,0], data_neu[:,1], '.', label = 'Data points', markersize = 1)
    ax3.plot(x_vals, f_lin(x_vals), label = 'Linear splines', lw = .5)
    ## Plot average for energies less than cutoff. It is close to .5
    ax3.plot(x_vals[~plotting_mask], f_cubic(x_vals[~plotting_mask]), x_vals[plotting_mask], \
        np.ones_like(x_vals[plotting_mask]) * f_cubic(x_vals[plotting_mask]).mean(), label = 'Cubic splines', lw = .5) 

   
    ax3.legend()
    fig3.tight_layout()

    plt.show()


if __name__ == '__main__':
    main()
