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
from scipy.special import sici, factorial
from iminuit import Minuit
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rcParams
from cycler import cycler

sys.path.append('Appstat2022\\External_Functions')
from ExternalFunctions import Chi2Regression, BinnedLH, UnbinnedLH
from ExternalFunctions import nice_string_output, add_text_to_ax    # Useful functions to print fit results on figure


## Change directory to current one
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

## Set plotting style and print options
sns.set_theme()
sns.set_style("darkgrid")
sns.set_context("paper") #Possible are paper, notebook, talk and poster

d = {'lines.linewidth': 2, 'axes.titlesize': 18, 'axes.labelsize': 18, 'xtick.labelsize': 12, 'ytick.labelsize': 12,\
     'legend.fontsize': 15, 'font.family': 'serif', 'figure.figsize': (9,6)}
d_colors = {'axes.prop_cycle': cycler(color = ['teal', 'navy', 'coral', 'plum', 'purple', 'olivedrab',\
         'black', 'red', 'cyan', 'yellow', 'khaki','lightblue'])}
rcParams.update(d)
rcParams.update(d_colors)
np.set_printoptions(precision = 5, suppress=1e-10)


### FUNCTIONS ----------------------------------------------------------------------------------

def get_statistics_from_fit(fitting_object, Ndatapoints, subtract_1dof_for_binning = False):
    
    Nparameters = len(fitting_object.values[:])
    if subtract_1dof_for_binning:
        Ndof = Ndatapoints - Nparameters - 1
    else:
        Ndof = Ndatapoints - Nparameters
    chi2 = fitting_object.fval
    prop = stats.chi2.sf(chi2, Ndof)
    return Ndof, chi2, prop

def do_chi2_fit(fit_function, x, y, dy, parameter_guesses, verbose = True):

    chi2_object = Chi2Regression(fit_function, x, y, dy)
    fit = Minuit(chi2_object, *parameter_guesses)
    fit.errordef = Minuit.LEAST_SQUARES

    if verbose:
        print(fit.migrad())
    else:
        fit.migrad()
    return fit

def do_LH_fit(fit_func, x_vals, paramters_guesses, bound, unbinned = True, bins = None, extended = True, verbose = True):
        if unbinned:
            LH_object = UnbinnedLH(fit_func, x_vals, bound = bound, extended = extended)
        else:
            LH_object = BinnedLH(fit_func, x_vals, bound = bound, bins = bins, extended = extended)
        
        fit = Minuit(LH_object, *paramters_guesses)
        fit.errordef = Minuit.LIKELIHOOD
        
        if verbose:
            print(fit.migrad())
        else:
            fit.migrad()
        return fit

def spherical_dot_prod(azi1, pol1, azi2, pol2):
        val = np.sin(pol1) * np.sin(pol2) * np.cos(azi2 - azi1) + np.cos(pol1) * np.cos(pol2)
        return val
    
def cum_auto_corr(angle_arr, cutoff):
    """angle has format [Npoints, (azimuth, polar)]
    """
    angles = angle_arr.astype('float')
    Npoints, _ = angles.shape
    val = np.empty([Npoints, Npoints])
    for i, ang in enumerate(angles):
        val[i,:] = np.sin(ang[1]) * np.sin(angles[:,1]) * np.cos(ang[0] -angles[:,0]) + np.cos(ang[1]) * np.cos(angles[:,1]) - cutoff
        val[i,i] = -1


    mask = (val > 0)
    sum = len(val[mask])
    multiplier = 1 / (Npoints * (Npoints - 1))
    return multiplier * sum


### MAIN ---------------------------------------------------------------------------------------

def P1():
    # find statistically compatable fit for each column. Use bootstrap for uncertainties
    # normalize
    # justify statistic compatibility
    # range for first col is 20-27. range for second is -1,1
    # Plot in each case.

    # use first 3 columns
    data = np.loadtxt('Exam_2023_Prob1.txt', usecols=(0,1,2))
    Npoints, _ = data.shape

  
    fig, ax = plt.subplots(ncols = 3)
    ax = ax.flatten()

    ranges = [(0,0), (0,0), (0,0)] 
    bin_widths = [None] * 3
    bins = [100, 100, 100]

    for i in np.arange(3):
        ranges[i] = (np.min(data[:,i]), np.max(data[:,i]))
        print(ranges[i][0])
        if i == 2:
            bins[i] = int(ranges[i][1] - ranges[i][0])
        bin_widths[i] = (ranges[i][1] - ranges[i][0]) / bins[i]

    
    histtype = ['stepfilled', 'stepfilled', 'bar']

    fits = [None] * 3

    norm4 = lambda x,a,b,c: - 1/a * np.cos(a*x) + c/b * np.exp(b *x) + x
  
    # Define functions and normalize
    f1 = lambda x,a,b: (1 + a * x + b * x **2) / (2 + 2 * b / 3)
    f2 = lambda x, a, b, c:   (1 + c * np.exp(b*x) + np.sin(a*x)) / (norm4(27,a,b,c) - norm4(20,a,b,c))

    f1_extended = lambda x,a,b,N: N * (1 + a * x + b * x **2) / (2 + 2 * b / 3)
    f2_extended = lambda x, a, b, c,N: N *  (1 + c * np.exp(b*x) + np.sin(a*x)) / (norm4(27,a,b,c) - norm4(20,a,b,c))

    # Already normalized, in the discrete sense
    pmf = lambda k, mean: stats.poisson.pmf(k, mu = mean)
    pmf_extended = lambda k, mean, N: N * pmf(k, mean)
    pmf_scaled = lambda k, mean: Npoints * pmf(k,mean)

    fit_func_extended = [f2_extended, f1_extended, pmf_extended]
    fit_func = [f2, f1, pmf]

    
    ## Minimize by Raster initially:
    logLH = lambda mu: - np.sum(np.log(pmf(data[:,2].astype('int'), mu)))
    logLH_vec = np.vectorize(logLH)
    mu = np.linspace(6,10,100)
    logLH_vals = logLH_vec(mu)
    logLH_min = np.min(logLH_vec(mu))
    mu_opt = mu[np.argmin(logLH_vec(mu))]
    print("Scan -logLH and lambda: ", logLH_min, mu_opt)

    # Define param guesses
    param_guess =  [np.array([4, -.3, 6000]), np.array([1,4]), np.array([15])]
 
    # Plot and fit
    for i, axx in enumerate(ax):
        ranges[i] = (np.min(data[:,i]), np.max(data[:,i]))
        counts, edges,_ =ax[i].hist(data[:,i], range = ranges[i], bins = bins[i], histtype = histtype[i], \
                                    alpha = .3, label = rf'Column {i+1}')
        
        bin_centers = (0.5 * (edges[1:] + edges[:-1]))
        dy = np.sqrt(counts)
        mask = (counts > 0)
        
        fits[i] = do_LH_fit(fit_func[i], data[:,i], param_guess[i], bound = ranges[i], extended = False)
        x_vals = np.linspace(ranges[i][0], ranges[i][1], 500)

        if i < 2:
            f_cumu = lambda xupper: integrate.quad(lambda x: fit_func[i](x, *fits[i].values[:]), ranges[i][0], xupper)[0]    
            f_cumu_vec = np.vectorize(f_cumu)
            print("KS test statistic and p value: ", stats.kstest(data[:,i], f_cumu_vec))

        if i < 2:    
            chi2 = np.sum((bin_widths[i] * Npoints * fit_func[i](bin_centers[mask], *fits[i].values[:]) - counts[mask]) ** 2 / dy[mask] ** 2)
            fit_vals = Npoints * bin_widths[i] * fit_func[i](x_vals, *fits[i].values[:])
            ax[i].plot(x_vals, fit_vals, label = rf'Fit {i+1}')
        else:
            chi2 = np.sum((pmf_scaled(bin_centers[mask].astype('int'), fits[i].values[:]) - counts[mask]) ** 2 / dy[mask] ** 2)
     
            ax[i].plot(bin_centers, pmf_scaled(bin_centers.astype('int'), fits[i].values[:]), label = rf'Fit {i+1}')

 
        Ndof = len(bin_centers[mask]) - 1 - len(fits[i].values[:]) # -1 cuz of normalization
        p_val = stats.chi2.sf(chi2, Ndof)
                 
        print(f'chi2 and pval for column {i+1}: ', chi2, p_val)
        ax[i].legend(loc = 'upper right', fontsize = 13)

 
        d = {'Ndof': Ndof, 'Chi2': chi2, 'p-val': p_val}
        text = nice_string_output(d, extra_spacing=2, decimals=2)
        add_text_to_ax(0.36, 0.75, text, ax[i], fontsize=11)

    fig.supxlabel('Value of independent variable', fontsize = 15)
    fig.supylabel('Count', fontsize = 15)
    fig.suptitle('Data histograms and fits', fontsize = 16)
    fig.tight_layout()
    plt.show()

def P2():
    # Quantify whether data is isotropically distributed

    data = np.loadtxt('Exam_2023_Problem2.txt', skiprows=2)

    azimuth = data[:, 0] # in Radians
    zenith = data[:,1] # in Radians
    Npoints = len(azimuth)

   
    Ntrials = 100
    Nresolution = 600
    cutoff_arr = np.linspace(-1, 1, Nresolution)
    KS_vals = []

    # Generate the autocorreleation function distribution for a high N to get an accurate approximation to true distribution
    data_distribution = np.empty(Nresolution)
    perfect_distribution = np.linspace(1,0,Nresolution)
    uniform = True
    if uniform:
        if 1:
            Nperfect = 10_000
            perfect_distribution = np.empty(Nresolution)
        
            azimuth = 2 * np.pi * (np.random.rand(Npoints))
            polar = np.arccos( 2 * (np.random.rand(Npoints)) - 1)
            points = np.r_['0,2', azimuth, polar].T
            for i, cutoff in enumerate(cutoff_arr):
                perfect_distribution[i] = cum_auto_corr(points, cutoff)
                data_distribution[i] = cum_auto_corr(data, cutoff)
        for i, cutoff in enumerate(cutoff_arr):
            data_distribution[i] = cum_auto_corr(data, cutoff)

        ks_data = stats.ks_2samp(perfect_distribution, data_distribution)[0]

        for n in np.arange(Ntrials):
            azimuth = 2 * np.pi * (np.random.rand(Npoints))
            polar = np.arccos( 2 * (np.random.rand(Npoints)) - 1)
            points = np.r_['0,2', azimuth, polar].T
    
            cum_vals = np.empty_like(cutoff_arr)
            for i, cutoff in enumerate(cutoff_arr):
                cum_vals[i] = cum_auto_corr(points, cutoff)
            #KS_vals.append(stats.ks_2samp(perfect_distribution, cum_vals)[0])
            KS_vals.append(stats.ks_2samp(np.linspace(1,0,Nresolution), cum_vals)[0])

        fig1, ax1 = plt.subplots()
        bin_width = 0.0015  
        range = (np.min(KS_vals), np.max(KS_vals))
        bins = int((range[1] - range[0]) / bin_width)

        if 1:
            significance_level = 0.95
            significance_index = int(significance_level * Ntrials)
            print(significance_index)
            KS_vals_sorted = np.sort(KS_vals)
            cutoff = KS_vals_sorted[significance_index]
            print(cutoff)

        count, edges, _ = ax1.hist(KS_vals, histtype='stepfilled', alpha=.4, bins = bins, lw = 1.5)
        ax1.plot([cutoff,cutoff], [0,np.max(count)/2], '--', label = rf'Decision boundary for $\alpha$ = {significance_level}')
        ax1.plot([ks_data,ks_data], [0,np.max(count)/2], label = rf'KS value for data')
        ax1.set(ylabel = 'Count', xlabel = 'KS Test statistic', title = 'KS test statistic distribution for sph. isotropic data')

        fig2,ax2=plt.subplots()
        ax2.plot(cutoff_arr, perfect_distribution, label = 'Isotropic')
        #ax2.hist(data_distribution, range = (-1,1), histtype='step', lw=1.5, label = 'Data', cumulative=-1,bins=Nresolution)
        ax2.plot(cutoff_arr, data_distribution,'--', label = 'Data')
        ax2.set(xlabel = rf'$\cos \phi$', ylabel = rf'Cumulative auto-correlation $C(\phi)$', title = rf'$C(\phi)$ for isotropic and data distribution')
        d = {'KS test statistic:': ks_data}
        text = nice_string_output(d, extra_spacing=2, decimals=2)
        add_text_to_ax(0.5, 0.75, text, ax2, fontsize=13)

        if 0:
            fig3 = plt.figure()
            ax3 = fig3.add_subplot(111, projection = 'mollweide')
            plt.grid(True, **dict (lw = 3))
            ax3.scatter(azimuth - np.pi, zenith - np.pi/2, marker = 'o', s = 3, alpha = .3)
            ax3.set(title = 'Mollweide projection of data points')

        ax2.legend()
        ax1.legend()
        fig1.tight_layout()
        plt.show()

def P3():
    ## data is given following the density below. Find the best fit values of a and b.
    # a,b are in [0,15]x[9,27]
    # Make a 2D raster scan of the test statistic used aronud the best fit params a and b

     

    data = np.loadtxt('Exam_2023_Prob3.txt')

    Npoints = len(data)
    range = (1, 3)
    bins = 200
    binwidth = (range[-1] - range[0]) / bins

    density = lambda x, a, b: np.cos(a * x) * np.cos(b * x) / x ** 2 + 2
    
    Si = lambda x: sici(x)[0]
    density_integral = lambda x, a, b: 0.5 * ((b - a) *  Si((a - b) * x) - (a + b) * Si((a + b) * x) \
                                              - 2 * np.cos(a*x) * np.cos(b*x) / x + 4 * x  )

    normalization_const = lambda a,b: density_integral(range[1], a, b) - density_integral(range[0], a, b)
    density_normed = lambda x, a, b: 1 / normalization_const(a,b) * density(x, a, b)

    
    # DO LH fit unbinned
    LH_unbinned_object = UnbinnedLH(density_normed, data, bound = range, extended = False)

    fit = Minuit(LH_unbinned_object, a = 5, b = 23)
    fit.errordef = Minuit.LIKELIHOOD
    print(fit.migrad())
    
    # PLOT fit
    fig, ax = plt.subplots()

    x_vals = np.linspace(range[0], range[1], 1500)
    density_scaled = lambda x: binwidth * Npoints * density_normed(x, *fit.values[:])
   
    ax.plot(x_vals, density_scaled(x_vals), label = 'LH fit')
    ax.set(xlabel = 'Elevation in meters', ylabel = 'Count', title ='Glacier elevation histogram and fit')

    d = {'a': [fit.values['a'], fit.errors['a']], 'b': [fit.values['b'], fit.errors['b']]}
    text = nice_string_output(d, extra_spacing=2, decimals=2)
    add_text_to_ax(0.55, 0.30, text, ax, fontsize=16)

    ax.hist(data, range = range, bins = bins, histtype='stepfilled', alpha = .3, label = 'Histogram of data')
    ax.legend()
    fig.tight_layout()


    ## DO raster scan around optimum values of a and b

    alpha_est, beta_est = fit.values['a'], fit.values['b']
    alpha_err, beta_err = fit.errors['a'], fit.errors['b']

    ## Essentially, we do a Raster scan around the est. values
    width_alpha, width_beta = 3, 3.5
    Nsubdivisions = 20
    alpha_vals = np.linspace(alpha_est - width_alpha, alpha_est + width_alpha, Nsubdivisions)
    beta_vals = np.linspace(beta_est - width_beta, beta_est + width_beta, Nsubdivisions)
  
    ALPHA, BETA = np.meshgrid(alpha_vals, beta_vals)
    logLH_max = -  fit.fval

    delta_logLH_vals_2_params = np.array([1.15, 3.09, 5.92])
    ### NBNBNBNB WRONG
    #delta_logLH_vals_2_params = np.array([0.5, 2, 4.5])



    func_norm_vec = lambda x:  density_normed(x, ALPHA, BETA)

    ## Calc logLH landscape
    logLH = np.zeros_like(ALPHA, dtype = 'float')

    for val in data:
        logLH = logLH + np.log(func_norm_vec(val))

    Raster_max = np.max(logLH)
    print("Raster logLH max: ", np.max(logLH))
    print("Min raster LLH diff: ", logLH_max - Raster_max)
    logLH_max = Raster_max

    fig0, ax0 = plt.subplots()
    alpha1 = np.argwhere(np.abs(logLH_max - logLH ) < delta_logLH_vals_2_params[0]) #NB: format is (beta,alpha)

    params_alpha = alpha_vals[alpha1[:,1]]
    params_beta = beta_vals[alpha1[:,0]]


    im1 = ax0.imshow(logLH, extent = (alpha_vals[0], alpha_vals[-1], beta_vals[0], beta_vals[-1]), origin= 'lower', vmin = -5220)
    ax0.set(xlabel = r'a', ylabel = r'b', title = r'Log likelihood landscape')

    divider = make_axes_locatable(ax0)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig0.colorbar(im1, cax=cax, orientation='vertical')
    ax0.plot(alpha_est, beta_est, 'x', lw = 4, markersize = 13)

    alpha1sig = np.argmax(logLH) 
    print(alpha1sig)
    ax0.plot(alpha1sig[1], alpha1sig[0], 'x', lw = 4.5, markersize = 12, label = 'Raster min')
    ax0.legend()

    if 0:
        alpha1sig = np.argwhere(np.abs(logLH - (logLH_max - delta_logLH_vals_2_params[0]) ) < .01) 
        params_alpha1= alpha_vals[alpha1sig[:,1]]
        params_beta1 = beta_vals[alpha1sig[:,0]]
        
    
        Ncontours = 3
        alpha_contour_vals = [None] * Ncontours
        beta_contour_vals = [None] * Ncontours

        for i, LH_val in enumerate(delta_logLH_vals_2_params):
            im = ax0.contour(ALPHA, BETA, logLH, levels = [logLH_max - delta_logLH_vals_2_params[i]], cmap = 'viridis', alpha = .6,\
                            label = rf'{i+1}$\sigma$ contour') #, 
            v = im.collections[0].get_paths()[0].vertices
            alpha_contour_vals[i] = v[:,0]
            beta_contour_vals[i] = v[:,1]


    
        ax0.plot(alpha_est, beta_est, 'x', lw = 4.5, markersize = 12)
        
        alpha1sig = np.argmax(LH) 
        ax0.plot(alpha1sig[1], alpha1sig[0], 'x', lw = 4.5, markersize = 12, label = 'Raster min')
        ax0.legend()
        alpha_1sig_int = [np.min(alpha_contour_vals[0]), np.max(alpha_contour_vals[0])]
        beta_1sig_int = [np.min(beta_contour_vals[0]), np.max(beta_contour_vals[0])]
        print("Raster scan conf int for alpha, beta: ", alpha_1sig_int, beta_1sig_int)
        print("Raster std alpha + and std alpha -: ", alpha_1sig_int[1] - alpha_est, alpha_est - alpha_1sig_int[0])
        print("Raster std beta + and std beta -: ", beta_1sig_int[1] - beta_est, beta_est - beta_1sig_int[0])




    plt.show()

def P4():
    # Fit double and triple gaussian to elevation data. Plot for each
    # HYP A: double gaussian. HYP B: triple gaussian
    # at what significane can hyp A be rejected compared to B? Ie wilks

    data = np.loadtxt('Exam_2023_Prob4.txt', skiprows=1) # units of meter
    Npoints = len(data)

    fig, ax = plt.subplots()

    bins = 100
    bounds = (np.min(data), np.max(data))
    binwidth = (bounds[1] - bounds[0]) / bins
    count, edges, _ = ax.hist(data, bins, range = bounds, histtype='stepfilled',alpha=.4)
    bin_centers = 0.5 * (edges[1:] + edges[:-1])
    dy = np.sqrt(count)

    def double_gaussian_LH(x, N1, N2, mean1, mean2, std1, std2):
        val1 = N1 * 1 / (np.sqrt(2 * np.pi) * std1) * np.exp(-0.5 * (x-mean1) ** 2 / std1 ** 2)
        val2 = N2 * 1 / (np.sqrt(2 * np.pi) * std2) * np.exp(-0.5 * (x-mean2) ** 2 / std2 ** 2)
        return val1 + val2

    def triple_gaussian_LH(x, N1, N2, N3, mean1, mean2, mean3, std1, std2, std3):
        val1 = N1 * 1 / (np.sqrt(2 * np.pi) * std1) * np.exp(-0.5 * (x-mean1) ** 2 / std1 ** 2)
        val2 = N2 * 1 / (np.sqrt(2 * np.pi) * std2) * np.exp(-0.5 * (x-mean2) ** 2 / std2 ** 2)
        val3 = N3 * 1 / (np.sqrt(2 * np.pi) * std3) * np.exp(-0.5 * (x-mean3) ** 2 / std3 ** 2)
        return val1 + val2 + val3

    def triple_gaussian_chi2(x, N1, N2, N3, mean1, mean2, mean3, std1, std2, std3):
        val1 = N1 * 1 / (np.sqrt(2 * np.pi) * std1) * np.exp(-0.5 * (x-mean1) ** 2 / std1 ** 2)
        val2 = N2 * 1 / (np.sqrt(2 * np.pi) * std2) * np.exp(-0.5 * (x-mean2) ** 2 / std2 ** 2)
        val3 = N3 * 1 / (np.sqrt(2 * np.pi) * std3) * np.exp(-0.5 * (x-mean3) ** 2 / std3 ** 2)
        return (val1 + val2 + val3 ) * binwidth


    def do_chi2_fit(fit_function, x, y, dy, parameter_guesses, verbose = True):

        chi2_object = Chi2Regression(fit_function, x, y, dy)
        fit = iminuit.Minuit(chi2_object, *parameter_guesses)
        fit.errordef = iminuit.Minuit.LEAST_SQUARES

        if verbose:
            print(fit.migrad())
        else:
            fit.migrad()
        return fit

 
    #Do chi2 fit to get approximate parameters for triple gaussian fit
  #  chi2_fit = do_chi2_fit(triple_gaussian_LH)


    ## Minimize with minuit for reference
    params3 = np.array([311,690, 900,3700, 5100,5600,380, 380,150])
    params2 = np.array([311,690,3700,5400,380,380])
   
    LH_unbinned_object2 = UnbinnedLH(double_gaussian_LH, data, bound = bounds, extended = True)
    LH_unbinned_object3 = UnbinnedLH(triple_gaussian_LH, data, bound = bounds, extended = True)

    fit2 = Minuit(LH_unbinned_object2, *params2)
    fit3 = Minuit(LH_unbinned_object3, *params3)

    fit2.errordef = Minuit.LIKELIHOOD
    fit3.errordef = Minuit.LIKELIHOOD
    print(fit2.migrad())
    print(fit3.migrad())
    
    x_vals = np.linspace(bounds[0], bounds[1], 1500)
    double_gaussian_scaled = lambda x:  binwidth * double_gaussian_LH(x, *fit2.values[:])
    triple_gaussian_scaled = lambda x:  binwidth * triple_gaussian_LH(x, *fit3.values[:])

    ax.plot(x_vals, double_gaussian_scaled(x_vals), label = 'Double Gaussian LH fit')
    ax.plot(x_vals, triple_gaussian_scaled(x_vals), label = 'Triple Gaussian LH fit')
    ax.set(xlabel = 'Elevation in meters', ylabel = 'Count', title ='Glacier elevation histogram and fits')


    wilks_chi2_val = - 2 * (- fit2.fval + fit3.fval)
    wilks_dof = 3
    p_val = stats.chi2.sf(wilks_chi2_val, wilks_dof)

    print("Prob. that double and triple Gaussian fits describe data equally well: ", p_val)



    ax.legend()
    fig.tight_layout()
    plt.show()

def P5a():
    # download file with test statistic values (booststrap)
    # wht is the critical value of the test statistic that corresponds to a one side p value of 4.55%?
    # if the true dsitrubtion is chi dsquare, does the boostrapped estimate mathch the expected value with k = 5?

    data = np.loadtxt("Exam_2023_Problem5a.txt")
    Npoints = len(data)

    # Find critical value   
    cutoff_fraction = 0.0455
    cutoff_index = Npoints - int(Npoints * cutoff_fraction)
    data_sorted = np.sort(data)

    crit_val = data_sorted[cutoff_index]

    print("sample critical value:  ", crit_val)

    ## find 4.55% critical value for k=5 chi2 dist
    crit_val_chi2 = stats.chi2.isf(cutoff_fraction, 5)

    print("expected cutoff for chi2 dist with 5 deg. of freedom: ", crit_val_chi2)
    print("p value corresponding to p = 4.55% for this distribution: ", stats.chi2.sf(crit_val, 5))




    fig, ax = plt.subplots()
    
    bins = 100
    range = (np.min(data), np.max(data))
    binwidth = (range[-1] - range[0]) / bins 
    print(binwidth)
    ## chi2f
    chi2_scaled = lambda x: binwidth * Npoints * stats.chi2.pdf(x, 5)
    x_vals = np.linspace(0, range[-1], 500)
    chi2_vals = chi2_scaled(x_vals)


    ax.hist(data, range = range, bins = bins, histtype = 'step', alpha = .3, lw = 1.5, label = 'Histogram of sample values')
    ax.plot([crit_val,crit_val], [0,10], label = '4.55 % cutoff')
    ax.plot(x_vals, chi2_vals, label = r'$\chi^2_{k=5}$ prob. density')
    ax.set(xlabel = 'Test statistic', ylabel = 'Count', title = 'Sample of test statistic distribution')
    ax.legend()
    plt.show()

def P5b():
    ## iterpolate following points using a lin and piecewise cubic Hermite
    ## what is the interploated y-value for x = 2 for each?
    x = [1, 1.7, 1.9, 2.2]
    y = [3.4, 3.9, 2.6, 3.1]
    Npoints = len(x)

    fig1, ax1 = plt.subplots()

    ## estimate dydx
    dx = np.diff(x)
    dy = np.diff(y)
    dydx = dy / dx

    # estimate the derivate of the endpoint as that of the second endpoint
    dydx = np.r_['0', dydx, dy[-1] / dx[-1]]

    ## estimate derivate of 2 central points a central differences
    dydx[1] = (y[2] - y[0]) / (dx[0] + dx[1])
    dydx[2] = (y[3] - y[1]) / (dx[1] + dx[2])
  
    ## Construct linear splines
    f_lin = interpolate.interp1d(x, y, kind = 'linear')
    f_cubic = interpolate.interp1d(x, y, kind = 'cubic')
    f_hermite = interpolate.CubicHermiteSpline(x, y, dydx, )

    ## Calc f(x=2)
    print("value of linear spline at x = 2:   ", f_lin(2))
    print("value of piecewise cubic Hermite spline at x = 2:   ", f_hermite(2))
    print("value of cubic spline at x = 2: ", f_cubic(2))

    # construct x-vals on which to evaluate the spline functions
    x_vals = np.linspace(np.min(x), np.max(x), 10 * Npoints)

    ax1.set(xlabel = 'x value', ylabel = 'y value', title = 'Datapoints and splines')
    ax1.plot(x, y, 'o', label = 'data points', markersize = 5)
    ax1.plot(x, f_lin(x), '-', label = 'Linear spline', lw = 1.7)
    ax1.plot(x_vals, f_hermite(x_vals), '--', label = 'PCHIP spline', lw = 1.7)
    ax1.plot(x_vals, f_cubic(x_vals), '--', label = 'Cubic spline', lw = 1.7)


    ax1.legend()
    fig1.tight_layout()
    plt.show()

def main():
    ## Set which problems to run
    p1, p2, p3, p4, p5a, p5b = True, False, False, False, False, False
    problem_numbers = [p1, p2, p3, p4, p5a, p5b]
    f_list = [P1, P2, P3, P4, P5a, P5b]

    for i, f in enumerate(f_list):
        if problem_numbers[i]:
            print(f'\nPROBLEM {i + 1}:')
            f()

if __name__ == '__main__':
    main()
