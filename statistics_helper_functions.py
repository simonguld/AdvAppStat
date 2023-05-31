# Author: Simon Guldager Andersen
# Date (latest update): 10-03-2023

### SETUP --------------------------------------------------------------------------------------------------------------------

## Imports:
import os, sys, time
import iminuit
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib import rcParams
from scipy import stats, integrate, optimize, constants, spatial

sys.path.append('Appstat2022\\External_Functions')
from ExternalFunctions import Chi2Regression, BinnedLH, UnbinnedLH
from ExternalFunctions import nice_string_output, add_text_to_ax    # Useful functions to print fit results on figure


### FUNCTIONS ----------------------------------------------------------------------------------------------------------------

def generate_dictionary(fitting_object, Ndatapoints, chi2_fit = True, chi2_suffix = None, subtract_1dof_for_binning = False):

    Nparameters = len(fitting_object.values[:])
    if chi2_suffix is None:
        dictionary = {'Entries': Ndatapoints}
    else:
        dictionary = {f'({chi2_suffix}) Entries': Ndatapoints}


    for i in range(Nparameters):
        dict_new = {f'{fitting_object.parameters[i]}': [fitting_object.values[i], fitting_object.errors[i]]}
        dictionary.update(dict_new)
    if subtract_1dof_for_binning:
        Ndof = Ndatapoints - Nparameters - 1
    else:
        Ndof = Ndatapoints - Nparameters
    if chi2_suffix is None:
        dictionary.update({f'Ndof': Ndof})
    else:
        dictionary.update({f'({chi2_suffix}) Ndof': Ndof})

    if chi2_fit:
        chi2 = fitting_object.fval
        p = stats.chi2.sf(chi2, Ndof)
        if chi2_suffix is None:
            dictionary.update({'Chi2': chi2, 'Prop': p})
        else:
            dictionary.update({f'({chi2_suffix}) Chi2': chi2, f'({chi2_suffix}) Prop': p})
    return dictionary

def calc_weighted_mean(x, dx):
    """
    returns: weighted mean, error on mean, Ndof, Chi2, p_val
    """
    assert(len(x) > 1)
    assert(len(x) == len(dx))
    
    var = 1 / np.sum(1 / dx ** 2)
    mean = np.sum(x / dx ** 2) * var

    # Calculate statistics
    Ndof = len(x) - 1
    chi2 = np.sum((x - mean) ** 2 / dx ** 2)
    p_val = stats.chi2.sf(chi2, Ndof)

    return mean, np.sqrt(var), Ndof, chi2, p_val

def calc_mean_std_sem(x, ddof = 1):
    """ returns mean, std, sem (standard error on mean)
    """
    Npoints = len(x)
    mean = x.mean()
    std = x.std(ddof = ddof)
    sem = std / np.sqrt(Npoints)
    return mean, std, sem

def calc_cov_matrix(data, ddof = 1):
    """assuming that each column represents a separate variable"""
    rows, cols = data.shape
    cov_matrix = np.empty([cols, cols])

    for i in range(cols):
        for j in range(i, cols):
            if ddof == 0:
                cov_matrix[i,j] = np.mean(data[:,i] * data[:,j]) - data[:,i].mean() * data[:,j].mean()
                cov_matrix[j,i] = cov_matrix[i,j]
            elif ddof == 1:
                cov_matrix[i,j] = 1/(rows - 1) * np.sum((data[:,i] - data[:,i].mean())*(data[:,j] - data[:,j].mean()))
                cov_matrix[j,i] = cov_matrix[i,j]
            else:
                print("The degrees of freedom must be 0 or 1")
                return None

    return cov_matrix

def calc_corr_matrix(x):
    """assuming that each column of x represents a separate variable"""
   
    data = x.astype('float')
    rows, cols = data.shape
    corr_matrix = np.empty([cols, cols])
 
    for i in range(cols):
        for j in range(i, cols):
                corr_matrix[i,j] = (np.mean(data[:,i] * data[:,j]) - data[:,i].mean() * data[:,j].mean()) / (data[:,i].std(ddof = 0) * data[:,j].std(ddof = 0))

        corr_matrix[j,i] = corr_matrix[i,j]
    return corr_matrix

def prop_err(dzdx, dzdy, x, y, dx, dy, correlation = 0):
    """ derivatives must takes arguments (x,y)
    """
    var_from_x = dzdx(x,y) ** 2 * dx ** 2
    var_from_y = dzdy (x, y) ** 2 * dy ** 2
    interaction = 2 * correlation * dzdx(x, y) * dzdy (x, y) * dx * dy

    prop_err = np.sqrt(var_from_x + var_from_y + interaction)

    if correlation == 0:
        return prop_err, np.sqrt(var_from_x), np.sqrt(var_from_y)
    else:
        return prop_err

def prop_err_3var(dzdx, dzdy, dzdt, x, y, t, dx, dy, dt):
    """ derivatives must takes arguments (x,y,t). Asummes no correlation between x,y,t
    """
    var_from_x = dzdx(x,y,t) ** 2 * dx ** 2
    var_from_y = dzdy (x, y, t) ** 2 * dy ** 2
    var_from_t = dzdt(x, y, t) ** 2 * dt ** 2

    prop_err = np.sqrt(var_from_x + var_from_y + var_from_t)

    return prop_err, np.sqrt(var_from_x), np.sqrt(var_from_y), np.sqrt(var_from_t)

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
    fit = iminuit.Minuit(chi2_object, *parameter_guesses)
    fit.errordef = iminuit.Minuit.LEAST_SQUARES

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
    
    fit = iminuit.Minuit(LH_object, *paramters_guesses)
    fit.errordef = iminuit.Minuit.LIKELIHOOD
    
    if verbose:
        print(fit.migrad())
    else:
        fit.migrad()
    return fit

def evaluate_likelihood_fit (fit_function, fmax, parameter_val_arr, log_likelihood_val, bounds, Ndatapoints, \
     Nsimulations, Nbins = 0, extended = True, unbinned = True):
    """
    fit_function is assumed to have the form f(x, *parameters), with x taking values in bounds
    Returns:: LL_values, p_value
    """
    LL_values = np.empty(Nsimulations)
    Nsucceses = 0
    max_iterations = 2 * Nsimulations
    iterations = 0
   
     # Simulate data
    while Nsucceses < Nsimulations and iterations < max_iterations:
        iterations += 1
   
        # Create values distributed according to fit_function

        x_vals, _, _ = rejection_sampling_uniform(lambda x: fit_function(x, *parameter_val_arr), fmax, bounds = bounds, Npoints = Ndatapoints, verbose = False)

        if 0:
            plt.figure()
            Nbins = 40
            plt.hist(x_vals, bins = Nbins, range = bounds)
            def gaussian_binned(x, N, mean, std):
                bin_width = (bounds[1] - bounds[0]) /  Nbins
                return N * bin_width * 1 / (np.sqrt(2 * np.pi) * std) * np.exp(-0.5 * (x-mean) ** 2 / std ** 2)

            xx = np.linspace(bounds[0], bounds[1], 500)

            plt.show()
          

        # Construct fitting object
        if unbinned:
            LLH_object = UnbinnedLH(fit_function, x_vals, bound = (bounds[0], bounds[1]), extended = extended)
            fit = iminuit.Minuit(LLH_object, *parameter_val_arr)
        else:
            LLH_object =  BinnedLH(fit_function, x_vals, bins = Nbins, bound = (bounds[0], bounds[1]), extended = extended)
            fit = iminuit.Minuit(LLH_object, *parameter_val_arr)

        fit.errordef = iminuit.Minuit.LIKELIHOOD
        fit.migrad()
        print(fit.fval)
        
        if 0:
            print("sim data points : ", len(x_vals))
            print("fit params: ", fit.values[:])
            plt.figure()
            Nbins = 50
            plt.hist(x_vals, bins = Nbins, range = bounds)
            def func_binned(x):
                bin_width = (bounds[1] - bounds[0]) /  Nbins
                return Ndatapoints * bin_width * fit_function(x, *fit.values[:])

            xx = np.linspace(bounds[0], bounds[1], 500)
            plt.plot(xx, func_binned(xx), 'r-')
            plt.show()

        if fit.fmin.is_valid:
            LL_values[Nsucceses] = fit.fval
            Nsucceses += 1
        else:
            print(f"ERROR: Fit did not converge for simulation no. {Nsucceses}. Log likelihood value is not collected.")

    mask = (LL_values > log_likelihood_val)
    p_value = len(LL_values[mask]) / Nsimulations

    return LL_values, p_value

def plot_likelihood_fits(LL_values, p_val, log_likelihood_val):
        Nsimulations = len(LL_values)
        fig0, ax0 = plt.subplots(figsize = (6,4))
        ax0.set_xlabel('Log likelihood value', fontsize = 18)
        ax0.set_ylabel('Count', fontsize = 18)
        ax0.set_title('Simulated log-likehood values', fontsize = 18)

        LL_std = LL_values.std(ddof = 1)
        counts, edges, _ = plt.hist(LL_values, bins = int(Nsimulations / 10), histtype = 'step', lw = 2, color = 'red');
        x_vals = 0.5 * (edges[:-1] + edges[1:])
        ax0.set_ylim(0,np.max(counts+5))
        ax0.plot([log_likelihood_val, log_likelihood_val], [0,np.max(counts)], 'k--', label = 'Log likelihood value (from fit)', lw = 2)

        ax00 = ax0.twinx()
        ax00.set_yticks(np.arange(0,1.1, 0.1))
        print("counts ",counts.sum())
        val_cumsum = np.cumsum(counts) / counts.sum()

        ax00.plot(x_vals, val_cumsum, 'k-', label = 'Cumulative distribution', lw = 2)
        # Adding fit results to plot:
        d = {'Entries':   Nsimulations,
            'Prob':     p_val}

        text = nice_string_output(d, extra_spacing=2, decimals=3)
        add_text_to_ax(0.05, 0.75, text, ax0, fontsize=16)

        fig0.legend( fontsize = 16, bbox_to_anchor = (0.25,0.65,0.25,0.25))
        fig0.tight_layout()
        return None

def one_sample_test(sample_array, exp_value, error_on_mean = None, one_sided = False, small_statistics = False):
    """ Assuming that the errors to be used are the standard error on the mean as calculated by the sample std 
    Returns test-statistic, p_val
    If a scalar sample is passed, the error on the mean must be passed as well, and large statistics is assumed
    """
    if np.size(sample_array) == 1:
        assert(error_on_mean is not None)
        assert(np.size(error_on_mean) == 1)
        assert(small_statistics == False)
        SEM = error_on_mean
        x = sample_array
    else:
        x = sample_array.astype('float')
        Npoints = np.size(x)
        SEM = x.std(ddof = 1) / np.sqrt(Npoints)
    
    test_statistic = (np.mean(x) - exp_value) / SEM

    if small_statistics:
        p_val = stats.t.sf(np.abs(test_statistic), df = Npoints - 1)
    else:
        p_val = stats.norm.sf(np.abs(test_statistic))

    if one_sided:
        return test_statistic, p_val
    else:
        return test_statistic, 2 * p_val

def two_sample_test(x, y, x_err = None, y_err = None, one_sided = False, small_statistics = False):
    """
    x,y must be 1d arrays of the same length. 
    If x and y are scalars, the errors on the means x_rr and y_rr must be passed as well, and small_statistics must be False
    If x and y are arrays, the standard errors on the mean will be used to perform the test

    Returns: test_statistics, p_val
    """
    Npoints = np.size(x)
    assert(np.size(x) == np.size(y))

    if x_err == None:
        SEM_x = x.std(ddof = 1) / np.sqrt(Npoints)
    else:
        assert(small_statistics == False)
        assert(np.size(x_err) == 1)
        SEM_x = x_err
        
    if y_err == None:
        SEM_y = y.std(ddof = 1) / np.sqrt(Npoints)
    else:
        assert(small_statistics == False)
        assert(np.size(y_err) == 1)
        SEM_y = y_err
        

    test_statistic = (np.mean(x) - np.mean(y)) / (np.sqrt(SEM_x ** 2 + SEM_y ** 2)) 

    if small_statistics:
        p_val = stats.t.sf(np.abs(test_statistic), df = 2 * (Npoints - 1))
    else:
        p_val = stats.norm.sf(np.abs(test_statistic))
    if one_sided:
        return test_statistic, p_val
    else:
        return test_statistic, 2 * p_val

def runstest(residuals):
   
    N = len(residuals)

    indices_above = np.argwhere(residuals > 0.0).flatten()
    N_above = len(indices_above)
    N_below = N - N_above

    print(N_above)
    print("bel", N_below)
    # calculate no. of runs
    runs = 1
    for i in range(1, len(residuals)):
        if np.sign(residuals[i]) != np.sign(residuals[i-1]):
            runs += 1

    # calculate expected number of runs assuming the two samples are drawn from the same distribution
    runs_expected = 1 + 2 * N_above * N_below / N
    runs_expected_err = np.sqrt((2 * N_above * N_below) * (2 * N_above * N_below - N) / (N ** 2 * (N-1)))

    # calc test statistic
    test_statistic = (runs - runs_expected) / runs_expected_err

    print("Expected runs and std: ", runs_expected, " ", runs_expected_err)
    print("Actual no. of runs: ", runs)
    # use t or z depending on sample size (2 sided so x2)
    if N < 50:
        p_val = 2 * stats.t.sf(np.abs(test_statistic), df = N - 2)
    else:
        p_val = 2 * stats.norm.sf(np.abs(test_statistic))

    return test_statistic, p_val

def calc_fisher_discrimminant(data1, data2, weight_normalization = 1):
    data_1 = data1.astype('float')
    data_2 = data2.astype('float')

    means_1 = np.sum(data_1, axis = 0)
    means_2 = np.sum(data_2, axis = 0)

    cov_1 = calc_cov_matrix(data_1, ddof = 1)
    cov_2 = calc_cov_matrix(data_2, ddof = 1)


    covmat_comb_inv = np.linalg.inv(cov_1 + cov_2)  
    weights = covmat_comb_inv @ (means_1 - means_2) / weight_normalization

    fisher_discrimminant_1 = np.sum((weights) * data_1, axis = 1) 
    fisher_discrimminant_2 = np.sum((weights) * data_2, axis = 1)
 
    return fisher_discrimminant_1, fisher_discrimminant_2, weights

def calc_ROC(hist1, hist2, signal_is_to_the_right_of_noise = True, input_is_hist = True, bins = None, range = None) :
    """
    This function is a modified version of code written by Troels Petersen
    Calculate ROC curve from two histograms (hist1 is signal, hist2 is background):
    if input_is_hist = False, the input entries are assumed to be arrays, in which case bins and range must be provided
    returns: False positive rate, True positive rate
    """
    # First we extract the entries (y values) and the edges of the histograms:
    # Note how the "_" is simply used for the rest of what e.g. "hist1" returns (not really of our interest)
    if input_is_hist:
        y_sig, x_sig_edges, _ = hist1 
        y_bkg, x_bkg_edges, _ = hist2
    else:
        y_sig, x_sig_edges = np.histogram(hist1, bins = bins, range = range)
        y_bkg, x_bkg_edges = np.histogram(hist2, bins = bins, range = range)
    

    # Check that the two histograms have the same x edges:
    if np.array_equal(x_sig_edges, x_bkg_edges) :
        
        # Extract the center positions (x values) of the bins (both signal or background works - equal binning)
        x_centers = 0.5*(x_sig_edges[1:] + x_sig_edges[:-1])
        
        # Calculate the integral (sum) of the signal and background:
        integral_sig = y_sig.sum()
        integral_bkg = y_bkg.sum()

        # Initialize empty arrays for the True Positive Rate (TPR) and the False Positive Rate (FPR):
        TPR = np.zeros_like(y_sig).astype('float') # True positive rate (sensitivity)
        FPR = np.zeros_like(y_sig).astype('float') # False positive rate ()
        
        # Loop over all bins (x_centers) of the histograms and calculate TN, FP, FN, TP, FPR, and TPR for each bin:
        for i, x in enumerate(x_centers): 
   
            # The cut mask
            cut = (x_centers < x)
            
            # True positive
            TP = np.sum(y_sig[~cut]) / integral_sig    # True positives
            FN = np.sum(y_sig[cut]) / integral_sig     # False negatives
        
            if TP == 0 and FN == 0:
                TPR[i] = 0
            else:
                TPR[i] = TP / (TP + FN)                    # True positive rate
          
            
            # True negative
            TN = np.sum(y_bkg[cut]) / integral_bkg      # True negatives (background)
            FP = np.sum(y_bkg[~cut]) / integral_bkg     # False positives
            if TN == 0 and FP == 0:
                FPR[i] = 0
            else:
                FPR[i] = FP / (FP + TN)                     # False positive rate   
        
        if signal_is_to_the_right_of_noise:
            return FPR, TPR
        else:
            # If hist2 is signal, TPR is actually FPR and the other way around
            return TPR, FPR
    
    else:
        AssertionError("Signal and Background histograms have different bins and/or ranges")

def calc_sample_purity(hist_signal, hist_background, numpy_hist = False, signal_is_to_the_right_of_noise = True) :
    """
    Big thanks to Troels for generously providing the code upon which this function is based
    """
    # First we extract the entries (y values) and the edges of the histograms:
    # Note how the "_" is simply used for the rest of what e.g. "hist1" returns (not really of our interest)
    if numpy_hist:
        y_sig, x_sig_edges = hist_signal
        y_bkg, x_bkg_edges = hist_background
    else:
        y_sig, x_sig_edges, _ = hist_signal
        y_bkg, x_bkg_edges, _ = hist_background

    if signal_is_to_the_right_of_noise is False:
        x_sig_edges *= - 1
        x_bkg_edges *= - 1
    
    # Check that the two histograms have the same x edges:
    if np.array_equal(x_sig_edges, x_bkg_edges) :
        
        # Extract the center positions (x values) of the bins (both signal or background works - equal binning)
        x_centers = 0.5*(x_sig_edges[1:] + x_sig_edges[:-1])
        
        # Calculate the integral (sum) of the signal and background:
        integral_sig = y_sig.sum()
        integral_bkg = y_bkg.sum()
    
        # Initialize empty arrays for the sample purity: TP/(TP+FP)
        SP = np.zeros_like(y_sig) # True positive rate (sensitivity)
        
        # Loop over all bins (x_centers) of the histograms and calculate TN, FP, FN, TP, FPR, and TPR for each bin:
        for i, x in enumerate(x_centers): 
            
            # The cut mask
            cut = (x_centers < x)
            
            # True positive
            sig_area = np.sum(y_sig[~cut])   # True positives
            bkg_area = np.sum(y_bkg[~cut])     # False positives
            if sig_area == 0 and bkg_area == 0:
                SP[i] = 0
            else:
                SP[i] = sig_area / (sig_area + bkg_area)                    # False positive rate     

        if signal_is_to_the_right_of_noise is False:
            x_centers *= -1       
        return x_centers, SP
    else:
        AssertionError("Signal and Background histograms have different bins and/or ranges")

def calc_ROC_AUC(FPR, TPR):
    """ 
    Calculate the area under and ROC curve. Takes false positive rate and true positive rate as parameters and return the area under the curve
    """
    area_under_curve = np.abs(np.trapz(TPR, FPR))
    return area_under_curve
