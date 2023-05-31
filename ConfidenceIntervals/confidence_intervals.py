# Author: Simon Guldager Andersen
# Date (latest update): 

### SETUP ------------------------------------------------------------------------------------

## Imports:
import os, sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats, integrate, optimize
from iminuit import Minuit
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cycler import cycler

sys.path.append('Appstat2022\\External_Functions')
from ExternalFunctions import Chi2Regression, BinnedLH, UnbinnedLH
from ExternalFunctions import nice_string_output, add_text_to_ax    # Useful functions to print fit results on figure


## Change directory to current one
os.chdir('AdvAppStat\ConfidenceIntervals')


### FUNCTIONS ----------------------------------------------------------------------------------

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

def gaussian_LH(x, mean, std):
    return  1 / (np.sqrt(2 * np.pi) * std) * np.exp(-0.5 * (x-mean) ** 2 / std ** 2)

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

def evaluate_likelihood_fit (fit_function, fmax, parameter_val_arr, log_likelihood_val, bounds, Ndatapoints, \
     Nsimulations, Nbins = 0, unbinned = True):
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
          

        # Construct fitting object
        if unbinned:
            LLH_object = UnbinnedLH(fit_function, x_vals, bound = (bounds[0], bounds[1]), extended = True)
            fit = Minuit(LLH_object, *parameter_val_arr)
        else:
            LLH_object =  BinnedLH(fit_function, x_vals, bins = Nbins, bound = (bounds[0], bounds[1]), extended = True)
            fit = Minuit(LLH_object, *parameter_val_arr)

        fit.errordef = Minuit.LIKELIHOOD
        fit.migrad()

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

        counts, edges, _ = plt.hist(LL_values, bins = 50, histtype = 'step', lw = 2, color = 'red');
        x_vals = 0.5 * (edges[:-1] + edges[1:])
        ax0.set_ylim(0,np.max(counts+5))
        ax0.plot([log_likelihood_val, log_likelihood_val], [0,np.max(counts)], 'k--', label = ' - Log likelihood value (from fit)', lw = 2)

        ax00 = ax0.twinx()
        ax00.set_yticks(np.arange(0,1.1, 0.1))
        val_cumsum = np.cumsum(counts) / counts.sum()

        ax00.plot(x_vals, val_cumsum, 'k-', label = 'Cumulative distribution', lw = 2)
        # Adding fit results to plot:
        d = {'Entries':   Nsimulations,
            'Prob':     p_val}

        text = nice_string_output(d, extra_spacing=2, decimals=3)
        add_text_to_ax(0.05, 0.75, text, ax0, fontsize=16)
        #ax00.legend(loc='best', fontsize=16); # could also be # loc = 'upper right' e.g.
        #ax00.legend(loc = 'best', fontsize = 16)
        fig0.legend( fontsize = 16, bbox_to_anchor = (0.25,0.65,0.25,0.25))
        fig0.tight_layout()
        return None


def get_statistics_from_fit(fitting_object, Ndatapoints, fixed_normalization = True):
    if fixed_normalization:
        Ndof = Ndatapoints - len(fitting_object.values[:])
    else:
        Ndof = Ndatapoints - 1 - len(fitting_object.values[:])
    chi2 = fitting_object.fval
    prop = stats.chi2.sf(chi2, Ndof)
    return Ndof, chi2, prop


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
 
    debug, exc1a_and_exc2, exc1b, exc1c, exc3,exc3_extra = False, True, False, False, False, False

    if debug:
        Nsamples = 200
        bounds = [-3,3]

        mu_true, sigma_true = 0.0,1
        param_guess = np.array([mu_true, sigma_true])


        func = lambda x, params: stats.norm.pdf(x, params[0], params[1])
        sample_values = stats.norm.rvs(mu_true, sigma_true, Nsamples)

        LH_object = UnbinnedLH(func, sample_values, bound = bounds, extended = False)
        fit = Minuit(LH_object, param_guess)
        fit.errordef = Minuit.LIKELIHOOD
     
        fit.migrad()
        if fit.fmin.is_valid is not True:
            print(f"LL-fit failed")
        minuit_er = fit.errors[:]
        minuit_val = fit.values[:]

        Nsimulations = 1000


        ## Raster scan
        delta_LLH = 1.15
       # delta_LLH = 0.5
        mu_width, sigma_width = 1, .8
        Nsubdivisions = 100
        mu_vals = np.linspace(mu_true - mu_width, mu_true + mu_width, Nsubdivisions)
        sigma_vals = np.linspace(sigma_true - sigma_width, sigma_true + sigma_width, Nsubdivisions)
        MU, SIG = np.meshgrid(mu_vals, sigma_vals)
      
        func = lambda x, mu, sig: stats.norm.pdf(x, mu, sig)
        func_vec = lambda x: func(x, MU, SIG)
        LLH_true = - fit.fval

        delta_LLH_arr = np.empty(Nsimulations)

        for i in np.arange(Nsimulations):

            sample_values = stats.norm.rvs(mu_true, sigma_true, Nsamples)
            LL_vals = np.zeros_like(MU)
            LLH_true = np.sum(np.log(func(sample_values, mu_true, sigma_true)))
            for val in sample_values:
                LL_vals = LL_vals + np.log(func_vec(val))
            delta_LLH_arr[i] =  LLH_true - np.max(LL_vals)

        ## Plot values
        chi2_LH = - 2 * delta_LLH_arr
        fig0, ax0 = plt.subplots()
        range = [0,20]
        bins = 50
        bin_width = (range[1] - range[0]) / bins
        count, edges,_ = ax0.hist(chi2_LH, range = range, bins = bins, histtype = 'stepfilled', alpha = .3, label = 'Sampled values')

        print(np.min(chi2_LH), np.max(chi2_LH), chi2_LH.mean())
        chi2 = lambda x, k: stats.chi2.pdf(x, k)

        x_vals = np.linspace(range[0], range[1], 1000)
        chi2_scaled =  lambda x, k : Nsimulations * bin_width * chi2(x, k)
        ax0.plot(x_vals, chi2_scaled(x_vals, 2), label = 'chi2 with Ndof = 2')

        bin_centers = 0.5 * (edges[1:] + edges[:-1])
        dy = np.sqrt(count)
        mask = (count > 0)
        chi2_obj = Chi2Regression(chi2_scaled, bin_centers[mask], count[mask], dy[mask])
        fit2 = Minuit(chi2_obj, k = 2)
        fit2.errordef = Minuit.LEAST_SQUARES
        fit2.migrad()

        fit_vals = chi2_scaled(x_vals, fit2.values['k'])
        ax0.plot(x_vals, fit_vals, label = f"chi2 with Ndof = {fit2.values['k']}")


        ax0.legend()
        plt.show()

         #   print("minuit and raster maxlnLH: ", fit.fval, "   ", np.max(LL_vals))
        if 0:
            conf_values = np.argwhere(np.abs(LL_vals - np.max(LL_vals)) < delta_LLH)
            params_mu = mu_vals[conf_values[:,1]]
            params_sigma = sigma_vals[conf_values[:,0]]

            print("Minuit err mu: ", minuit_er[0])
            print("Minuit err  sigma: ", minuit_er[1])
            print("clean raster mu int :", np.abs(minuit_val[0] -np.min(params_mu)), np.abs(minuit_val[0]-np.max(params_mu)))
            print("clean raster sigma int :", np.abs(minuit_val[1]-np.min(params_sigma)), np.abs(minuit_val[1]-np.max(params_sigma)))

    if exc1a_and_exc2:
        ## EXC 1: Sim. values acc to pdf and fit estimators each time and plot distribution
        Npoints = 2000
        Nsimulations = 200
        bounds = [-0.95, 0.95]
        alpha_true, beta_true = 0.5, 0.5

        params_est_list = np.empty([2, Nsimulations])
        params_err_list = np.empty_like(params_est_list)

        def func(x, alpha, beta):
            return np.maximum(1e-10, 1 + alpha * x + beta * np.power(x, 2))

        norm_const = lambda lower_bound, upper_bound, alpha, beta: upper_bound - lower_bound \
            + alpha/2 * (upper_bound ** 2 - lower_bound ** 2) + beta/3 * (upper_bound ** 3 - lower_bound ** 3)
        
        func_norm = lambda x, alpha, beta: 1 / (norm_const(bounds[0], bounds[1], alpha, beta)) * func(x, alpha, beta)
 
        fmax = max(func_norm(bounds[0], alpha_true, beta_true), func_norm(bounds[1], alpha_true, beta_true), -alpha_true / (2 * beta_true))


        for i in np.arange(Nsimulations):

            sample_values = rejection_sampling_uniform(lambda x: func_norm(x, alpha_true, beta_true),\
                fmax, bounds, Npoints, verbose = False)[0]

            ## Minimize with minuit for reference
            LH_unbinned_object = UnbinnedLH(func_norm, sample_values, bound = bounds, extended = False)
            fit = Minuit(LH_unbinned_object, alpha = alpha_true, beta = beta_true)
            fit.errordef = Minuit.LIKELIHOOD
            fit.migrad()
            if fit.fmin.is_valid is not True:
                print(f"LL-fit failed for simulation {i}")
            params_est_list[:,i] = fit.values[:]
            params_err_list[:,i] = fit.errors[:]
        
      
        ## Find 1sigma confidence levels. Frac NOT contained in 1sig central interval
        crit_val_lower = stats.norm.cdf(-1, loc = 0, scale = 1)
        crit_val_upper = stats.norm.cdf(1, loc = 0, scale = 1)
        # Sort parameter lists
        alpha_list = np.sort(params_est_list[0,:])
        beta_list = np.sort(params_est_list[1,:])

        left_index = int(np.ceil(crit_val_lower * len(alpha_list) ))
        right_index = int(np.ceil(crit_val_upper * len(alpha_list) ))
        #print("Critical 1sigma indices: ", left_index, right_index)

        conf_int_alpha = [alpha_list[left_index], alpha_list[right_index]]
        conf_int_beta = [beta_list[left_index], beta_list[right_index]]
        conf_ints = [conf_int_alpha, conf_int_beta]
        alpha_up_down  = np.abs(conf_int_alpha - alpha_list.mean())
        beta_up_down = np.abs(conf_int_beta - beta_list.mean())
        print("1sig interval for alpha: ", conf_int_alpha)
        print("1 sig interval for beta: ", conf_int_beta)
        print("alpha upper lower dist from min: ", np.abs(conf_int_alpha - alpha_list.mean()))
        print("beta upper lower dist from min: ", np.abs(conf_int_beta - beta_list.mean()))

        fig1, ax1 = plt.subplots(ncols = 2, figsize = (16,9))
        ax1 = ax1.flatten()
       
        range_alpha = (np.min(params_est_list[0,:]), np.max(params_est_list[0,:]))
        range_beta = (np.min(params_est_list[1,:]), np.max(params_est_list[1,:]))
        bins = 50
        ranges = [range_alpha, range_beta]
        names = ['Alpha', 'Beta']
        means = [np.mean(params_est_list[0,:]), np.mean(params_est_list[1,:])]
        stds = [params_est_list[0,:].std(ddof = 1), params_est_list[1,:].std(ddof = 1)]

        d = {'Entries': Nsimulations}
        print('Av. minuit err of alpha and beta: ', params_err_list[0,:].mean(), params_err_list[1,:].mean())
       # print("Std of distribution as err estimate: ", stds[0], stds[1])
        
        for i, ax in enumerate(ax1):
            count, _, _ = ax.hist(params_est_list[i,:], range = ranges[i], bins = bins, \
                histtype = 'stepfilled', alpha = .6, lw =2, label = 'Fitted ML values')
            ax.set(xlabel = f'{names[i]}', ylabel = 'count', title = 'simulated ML estimates')
            ax.plot([means[i], means[i]], [0, np.max(count)], label = 'Sample mean')
            ax.plot([conf_ints[i][0], conf_ints[i][0]], [0, np.max(count)], label = 'Lower 1sig conf bound')
            ax.plot([conf_ints[i][1], conf_ints[i][1]], [0, np.max(count)], label = 'Upper 1sig conf bound')

            d0 = {'Mean': means[i], 'Lower 1sig conf bound': conf_ints[i][0], 'Upper 1 sig conf bound': conf_ints[i][1]}
            d.update(d0)
            text = nice_string_output(d, extra_spacing=2, decimals=3)
            add_text_to_ax(0.05, 0.75, text, ax, fontsize=13)
            ax.legend()
        
        fig1.tight_layout()

        if 0:
            fig3, ax3 = plt.subplots()
            bins2d = (14,10)
            ax3.scatter(params_est_list[0,:], params_est_list[1,:], alpha = .5, s = 15)
            corr = np.corrcoef(params_est_list[0,:], params_est_list[1,:])[0,1]
            text = nice_string_output({'Pearson corr ': corr}, extra_spacing=2, decimals=3)
            add_text_to_ax(0.05, 0.75, text, ax3, fontsize=13)
            ax3.set(xlabel = 'alpha', ylabel = 'beta')



        ## EXC 2: Sim 2000 values acc to alpha = beta = .5. Plot lnLH contours related to 1 sigma, 2sigma and 3 sigma. Also est uncertainties
        # Just use the last sample and fit values

        sample_values = rejection_sampling_uniform(lambda x: func_norm(x, alpha_true, beta_true),\
                fmax, bounds, Npoints, verbose = False)[0]

        ## Minimize with minuit for reference
        LH_unbinned_object = UnbinnedLH(func_norm, sample_values, bound = bounds, extended = False)
        fit = Minuit(LH_unbinned_object, alpha = alpha_true, beta = beta_true)
        fit.errordef = Minuit.LIKELIHOOD
        fit.migrad()


        alpha_est, beta_est = fit.values['alpha'], fit.values['beta']
        alpha_err, beta_err = fit.errors['alpha'], fit.errors['beta']

        ## Essentially, we do a Raster scan around the est. values
        width_alpha, width_beta = 0.5, 0.5
        Nsubdivisions = 100
        alpha_vals = np.linspace(alpha_est - width_alpha, alpha_est + width_alpha, Nsubdivisions)
        beta_vals = np.linspace(beta_est - width_beta, beta_est + width_beta, Nsubdivisions)
        #alpha_vals = np.linspace(0,1, Nsubdivisions)
        #beta_vals = np.linspace(0,1, Nsubdivisions)

        ALPHA, BETA = np.meshgrid(alpha_vals, beta_vals)
        logLH_max = -  fit.fval

        delta_logLH_vals_2_params = np.array([1.15, 3.09, 5.92])
        ### NBNBNBNB WRONG
        #delta_logLH_vals_2_params = np.array([0.5, 2, 4.5])

        print("\nFOR RASTER")
       # print("\alpha beta est and logLH: ", alpha_est, beta_est, logLH_max)
        print("alpha beta minuit error: ", alpha_err, beta_err)


        func_norm_vec = lambda x: 1 / (norm_const(bounds[0], bounds[1], ALPHA, BETA)) * func(x, ALPHA, BETA)

        ## Calc logLH landscape
        logLH = np.zeros_like(ALPHA, dtype = 'float')

        for val in sample_values:
            logLH = logLH + np.log(func_norm_vec(val))

        Raster_max = np.max(logLH)
       # print("Raster logLH max: ", np.max(logLH))
        print("Min raster LLH diff: ", logLH_max - Raster_max)
        logLH_max = Raster_max

        fig0, ax0 = plt.subplots()
        alpha1 = np.argwhere(np.abs(logLH_max - logLH ) < delta_logLH_vals_2_params[0]) #NB: format is (beta,alpha)
   
        params_alpha = alpha_vals[alpha1[:,1]]
        params_beta = beta_vals[alpha1[:,0]]
        ax0.plot([alpha_est - alpha_up_down[0],alpha_est - alpha_up_down[0]], [beta_vals[0], beta_vals[-1]], 'k')
        ax0.plot([alpha_est + alpha_up_down[1],alpha_est + alpha_up_down[1]], [beta_vals[0], beta_vals[-1]], 'k')
        ax0.plot([alpha_vals[0],alpha_vals[-1]], [beta_est - beta_up_down[0],beta_est - beta_up_down[0]], 'k')
        ax0.plot([alpha_vals[0],alpha_vals[-1]], [beta_est + beta_up_down[1],beta_est + beta_up_down[1]], 'k')
        ax0.grid(False)
      #  print("clean raster alpha int :", np.min(params_alpha), np.max(params_alpha))
       # print("clean raster beta int :", np.min(params_beta), np.max(params_beta))
     #   print(np.min(np.sort(params_beta)))
      #  print(np.max(logLH_max - logLH[alpha1[:,0],alpha1[:,1]]))
        #print(logLH_max - np.max(logLH[np.min(params_beta),:]))
        #print(logLH[alpha_min] - logLH_max, logLH[alpha_max] - logLH_max)
        im1 = ax0.imshow(logLH, extent = (alpha_vals[0], alpha_vals[-1], beta_vals[0], beta_vals[-1]), origin= 'lower', vmin = logLH_max-18)
     

        alpha1sig = np.argwhere(np.abs(logLH - (logLH_max - delta_logLH_vals_2_params[0]) ) < .01) 
        params_alpha1= alpha_vals[alpha1sig[:,1]]
        params_beta1 = beta_vals[alpha1sig[:,0]]
       
      #  fig00, ax00 = plt.subplots()
      #  print(len(params_alpha1))
      #  print(len(params_beta1))
       # print(np.min(np.sort(params_alpha1)))
        #print(np.min(np.sort(params_beta1)))
        ax0.scatter(params_alpha, params_beta, label = 'fun', s = 5)
    #    ax00.scatter(params_alpha1, params_beta1)
        #plot contours and extraxt contour points
        Ncontours = 3
        alpha_contour_vals = [None] * Ncontours
        beta_contour_vals = [None] * Ncontours

        for i, LH_val in enumerate(delta_logLH_vals_2_params):
            im = ax0.contour(ALPHA, BETA, logLH, levels = [logLH_max - delta_logLH_vals_2_params[i]], cmap = 'viridis', alpha = .6) #, 
            v = im.collections[0].get_paths()[0].vertices
            alpha_contour_vals[i] = v[:,0]
            beta_contour_vals[i] = v[:,1]

     #  ax00.scatter(alpha_contour_vals[0], beta_contour_vals[0], label = 'imshow')
    #   ax00.legend()
        ## Now extract min and max values of alpha and beta to obtain 1sig conf interval
        alpha_1sig_int = [np.min(alpha_contour_vals[0]), np.max(alpha_contour_vals[0])]
        beta_1sig_int = [np.min(beta_contour_vals[0]), np.max(beta_contour_vals[0])]
        print("Raster scan conf int for alpha, beta: ", alpha_1sig_int, beta_1sig_int)
        print("Raster std alpha + and std alpha -: ", alpha_1sig_int[1] - alpha_est, alpha_est - alpha_1sig_int[0])
        print("Raster std beta + and std beta -: ", beta_1sig_int[1] - beta_est, beta_est - beta_1sig_int[0])


        if 0:
            divider = make_axes_locatable(ax0)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig0.colorbar(im, cax=cax, orientation='vertical')

        if 1:
            divider = make_axes_locatable(ax0)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig0.colorbar(im1, cax=cax, orientation='vertical')

        ax0.plot(alpha_est, beta_est, 'x', lw = 2.5, markersize = 12)
        ax0.set(xlabel = r'$\alpha$', ylabel = r'$\beta$', title = r'$\ln LH$')

    
        plt.show()

    if exc1b:
        ## EXC2:
        # 1) Parametric bootstrapping. Fit 'data' once to obtain est. for alpha, beta
        # --> to obtain uncertainty est, gen. Nsimulations data sets acc. to these estimates, and fit for alpha,beta each time
        ## Plot distribution. Conf. intervals for alpha beta can now be 'counted' without assuming Gaussian distribution
        # 2) Repeat, but now:
        # fix alpha = .5 and only fit beta. What happens to the conf interval of beta?
        # 3) repeat with new range [-.9,0.85]


        ## PART 1
        Npoints = 2000
        Nsimulations = 100
        bounds = [-0.95, 0.95]
        alpha_true, beta_true = .5, .5
   
        params_est_list = np.empty([2, Nsimulations])
        params_err_list = np.empty_like(params_est_list)

        def func(x, alpha, beta):
            return np.maximum(1e-10, 1 + alpha * x + beta * np.power(x, 2))

        norm_const = lambda lower_bound, upper_bound, alpha, beta: upper_bound - lower_bound \
            + alpha/2 * (upper_bound ** 2 - lower_bound ** 2) + beta/3 * (upper_bound ** 3 - lower_bound ** 3)
     
        func_norm = lambda x, alpha, beta: 1 / (norm_const(bounds[0], bounds[1], alpha, beta)) * func(x, alpha, beta)
        fmax = func_norm(bounds[1], alpha_true, beta_true)
        
        ## RUN once to find est of alpha_opt and beta_opt
        sample_values = rejection_sampling_uniform(lambda x: func_norm(x, alpha_true, beta_true),\
                    fmax, bounds, Npoints, verbose = False)[0]

        ## Minimize with minuit for reference
        LH_unbinned_object = UnbinnedLH(func_norm, sample_values, bound = bounds, extended = False)
        fit0 = Minuit(LH_unbinned_object, alpha = alpha_true, beta = beta_true)
        fit0.errordef = Minuit.LIKELIHOOD
        fit0.migrad()
        alpha_orig_est, beta_orig_est = fit0.values['alpha'], fit0.values['beta']
        print("alpha, beta orig estimates: ", alpha_orig_est, beta_orig_est)

        for i in np.arange(Nsimulations):
            fmax = func_norm(bounds[1], alpha_orig_est, beta_orig_est)
            sample_values = rejection_sampling_uniform(lambda x: func_norm(x, alpha_orig_est, beta_orig_est),\
                fmax, bounds, Npoints, verbose = False)[0]

            ## Minimize with minuit for reference
            LH_unbinned_object = UnbinnedLH(func_norm, sample_values, bound = bounds, extended = False)
            fit = Minuit(LH_unbinned_object, alpha = alpha_orig_est, beta = beta_orig_est)
            fit.errordef = Minuit.LIKELIHOOD
            fit.migrad()
            # print(fit.values[:])
            if fit.fmin.is_valid is not True:
                print(f"LL-fit failed for simulation {i}")
            params_est_list[:,i] = fit.values[:]
            params_err_list[:,i] = fit.errors[:]


        ## Find 1sigma confidence levels. Frac NOT contained in 1sig central interval
        crit_val_lower = stats.norm.cdf(-1, loc = 0, scale = 1)
        crit_val_upper = stats.norm.cdf(1, loc = 0, scale = 1)
        # Sort parameter lists
        alpha_list = np.sort(params_est_list[0,:])
        beta_list = np.sort(params_est_list[1,:])

        left_index = int(np.ceil(crit_val_lower * len(alpha_list) ))
        right_index = int(np.ceil(crit_val_upper * len(alpha_list) ))
        print("Critical 1sigma indices: ", left_index, right_index)

        conf_int_alpha = [alpha_list[left_index], alpha_list[right_index]]
        conf_int_beta = [beta_list[left_index], beta_list[right_index]]
        conf_ints = [conf_int_alpha, conf_int_beta]

        print("1sig interval for alpha: ", conf_int_alpha)
        print("1 sig interval for beta: ", conf_int_beta)

        fig4, ax4 = plt.subplots(ncols = 2, figsize = (16,9))
        ax4 = ax4.flatten()
     
        range_alpha = (np.min(params_est_list[0,:]), np.max(params_est_list[0,:]))
        range_beta = (np.min(params_est_list[1,:]), np.max(params_est_list[1,:]))

        bins = 50

        ranges = [range_alpha, range_beta]
        names = ['Alpha', 'Beta']
        means = [np.mean(params_est_list[0,:]), np.mean(params_est_list[1,:])]
        stds = [params_est_list[0,:].std(ddof = 1), params_est_list[1,:].std(ddof = 1)]

        d = {'Entries': Nsimulations}


        print('Av. minuit err of alpha and beta: ', params_err_list[0,:].mean(), params_err_list[1,:].mean())
        print("Std of distribution as err estimate: ", stds[0], stds[1])
        
        for i, ax in enumerate(ax4):
            count, _, _ = ax.hist(params_est_list[i,:], range = ranges[i], bins = bins, \
                histtype = 'stepfilled', alpha = .6, lw =2, label = 'Fitted ML values')
            ax.set(xlabel = f'{names[i]}', ylabel = 'count', title = 'simulated ML estimates')
            ax.plot([means[i], means[i]], [0, np.max(count)], label = 'Sample mean')
            ax.plot([conf_ints[i][0], conf_ints[i][0]], [0, np.max(count)], label = 'Lower 1sig conf bound')
            ax.plot([conf_ints[i][1], conf_ints[i][1]], [0, np.max(count)], label = 'Upper 1sig conf bound')
            d0 = {'Mean': means[i], 'Std': stds[i]}
            d.update(d0)
            text = nice_string_output(d, extra_spacing=2, decimals=3)
            add_text_to_ax(0.05, 0.75, text, ax, fontsize=13)
            ax.legend()
        
        fig4.tight_layout()


        # 2) Repeat, but fix alpha = .5 and only fit beta. What happens to the conf interval of beta?
        # 3) repeat but with range = [-.9,.85]
        part2, part3 = False, True
        
        Npoints = 2000
        Nsimulations = 500
        if part2:
            bounds = [-0.95, 0.95]
           
        elif part3:
            bounds = [-0.9, 0.85]
        print("\nKeeping alpha=0.5 constant and using bounds = ", bounds)

        alpha_true, beta_true = .5, .5
   
        params_est_list = np.empty(Nsimulations)
        params_err_list = np.empty_like(params_est_list)

        func_norm = lambda x, beta: 1 / (norm_const(bounds[0], bounds[1], alpha_true, beta)) * func(x, alpha_true, beta)
        fmax = func_norm(bounds[1], beta_true)
        
        ## RUN once to find est of alpha_opt and beta_opt
        sample_values0 = rejection_sampling_uniform(lambda x: func_norm(x, beta_true),\
                    fmax, bounds, Npoints, verbose = False)[0]

        ## Minimize with minuit for reference
        LH_unbinned_object = UnbinnedLH(func_norm, sample_values0, bound = bounds, extended = False)
        fit0 = Minuit(LH_unbinned_object, beta = beta_true)
        fit0.errordef = Minuit.LIKELIHOOD
        fit0.migrad()
        beta_orig_est = fit0.values['beta']
        print("beta orig estimate and -LL val: ", beta_orig_est, fit0.fval)

        for i in np.arange(Nsimulations):
            fmax = func_norm(bounds[1], beta_orig_est)
            sample_values = rejection_sampling_uniform(lambda x: func_norm(x, beta_orig_est),\
                fmax, bounds, Npoints, verbose = False)[0]

            ## Minimize with minuit for reference
            LH_unbinned_object = UnbinnedLH(func_norm, sample_values, bound = bounds, extended = False)
            fit = Minuit(LH_unbinned_object, beta = beta_orig_est)
            fit.errordef = Minuit.LIKELIHOOD
            fit.migrad()
            # print(fit.values[:])
            if fit.fmin.is_valid is not True:
                print(f"LL-fit failed for simulation {i}")
            params_est_list[i] = fit.values['beta']
            params_err_list[i] = fit.errors['beta']


        ## Find 1sigma confidence levels. Frac NOT contained in 1sig central interval
        crit_val_lower = stats.norm.cdf(-1, loc = 0, scale = 1)
        crit_val_upper = stats.norm.cdf(1, loc = 0, scale = 1)
        # Sort parameter lists
        beta_list = np.sort(params_est_list)

        left_index = int(np.ceil(crit_val_lower * len(beta_list)))
        right_index = int(np.ceil(crit_val_upper * len(beta_list)))
        print("Critical 1sigma indices: ", left_index, right_index)


        conf_int_beta = [beta_list[left_index], beta_list[right_index]]


        print("1 sig interval for beta, keeping alpha = .5 const: ", conf_int_beta)

        fig5, ax5 = plt.subplots()
  
        range_beta = (np.min(params_est_list), np.max(params_est_list))

        bins = 50
        names = 'Beta'
        means = np.mean(params_est_list)
        stds = params_est_list.std(ddof = 1)

        d = {'Entries': Nsimulations}

        print('Av. minuit err of beta: ',  params_err_list.mean())
        print("Std of distribution as err estimate: ", stds)
        
   
        count, _, _ = ax5.hist(params_est_list, range = range_beta, bins = bins, \
            histtype = 'stepfilled', alpha = .6, lw =2, label = 'Fitted ML values')
        ax5.set(xlabel = f'{names}', ylabel = 'count', title = 'simulated ML estimates')
        ax5.plot([means, means], [0, np.max(count)], label = 'Sample mean')
        ax5.plot([conf_int_beta[0], conf_int_beta[0]], [0, np.max(count)], label = 'Lower 1sig conf bound')
        ax5.plot([conf_int_beta[1], conf_int_beta[1]], [0, np.max(count)], label = 'Upper 1sig conf bound')
        d0 = {'Mean': means, 'Std': stds}
        d.update(d0)
        text = nice_string_output(d, extra_spacing=2, decimals=3)
        add_text_to_ax(0.05, 0.75, text, ax5, fontsize=13)
        ax5.legend()
        
        fig5.tight_layout()


        if exc1c:
            ## EXC 1C:
            # Again using range [-0.9,0.85], 2000 MC values and keeping alpha const, (i.e. same ML estimator for beta)
            # use the likelihood scan approach to calc uncertainty for beta
            # the uncertanity on beta is the amount beta needs to change to decrease lnLL with 0.5 (since 1 param)
            # Check if the conf int. is sym. around central point
            
            deltaLL = 0.5
            shift = 0.25
            beta_range = [beta_orig_est - shift, beta_orig_est + shift]
            Nsubdivisions = 1000
            beta_vals = np.linspace(beta_range[0], beta_range[1], Nsubdivisions)


            func_norm_vec = lambda x: 1 / (norm_const(bounds[0], bounds[1], alpha_true, beta_vals)) * func(x, alpha_true, beta_vals)

            logLL = np.zeros_like(beta_vals)
      
            for val in sample_values0:
                logLL += np.log(func_norm_vec(val))


            logLL_max = np.max(logLL)
            logLL_max_index = np.argmax(logLL)

  
            crit_val_lower_index = np.argmin(np.abs(logLL[:logLL_max_index] - (logLL_max - deltaLL))).astype('int')
            crit_val_upper_index = np.argmin(np.abs(logLL[logLL_max_index:] - (logLL_max - deltaLL))) + logLL_max_index
            LL_crit_vals = [logLL[crit_val_lower_index], logLL[crit_val_upper_index]]
            beta_crit_vals = [beta_vals[crit_val_lower_index], beta_vals[crit_val_upper_index]]
            print("max lnLH val, beta val: ", logLL_max, beta_vals[logLL_max_index])
            print("Critical lower and upper lnLH values: ", logLL[crit_val_lower_index], logLL[crit_val_upper_index])
            print("Bootstrap beta 1 sig conf int.", [beta_vals[crit_val_lower_index], beta_vals[crit_val_upper_index]])
            print("Corresponding (lnLH_max - lnLH_critical) values: ", logLL_max - logLL[crit_val_lower_index], logLL_max - logLL[crit_val_upper_index])
            print("sigma beta +: ", beta_vals[crit_val_upper_index] - beta_orig_est)
            print("sigma beta -: ", beta_orig_est - beta_vals[crit_val_lower_index])
            fig6, ax6 = plt.subplots()

            ax6.plot(beta_vals, logLL, '.-')
            ax6.set(xlabel = r'$\beta$', ylabel = r'$\ln LH$')
            ax6.plot([beta_crit_vals[0], beta_crit_vals[0]], [np.min(logLL), logLL[crit_val_lower_index]], 'k-', label = r'$\hat{\beta} - \sigma_{\beta}$')
            ax6.plot([beta_vals[logLL_max_index], beta_vals[logLL_max_index]], [np.min(logLL), logLL_max], 'k-', label = r'$\hat{\beta}$')
            ax6.plot([beta_crit_vals[1], beta_crit_vals[1]], [np.min(logLL), logLL[crit_val_upper_index]], 'k-', label = r'$\hat{\beta} + \sigma_{\beta}$')
            ax6.legend()

        plt.show()

    if exc3:
        ## Download file with 2 pseudotrials of the function used already. Estimate alpha, beta
        ## Estimate p-val using LL-p-val-calculator and compare to chi^2 val

        # Option 1: Fit each pseudo exp. and average
        # Option 2: Treat all sample values as stemming from one pseudo experiment


        ## TO DO
        # 1) Chi2 fit both distributions and see what's up
        # 2) Do extras
        ### WHEN TIME
        # 3) Få LL-p-val ting til at spinde som en missetat
        # 4) Do LogLL extras



        sample_values = np.loadtxt('MLE_Variance_data.txt')
        sample_values_copy = sample_values.T.astype('float')
        sample_values = sample_values.flatten()


        Npoints = len(sample_values)
  
        bounds = [-1, 1]
        alpha_guess = .5
        beta_guess = .5
       
      
        def func(x, alpha, beta):
            return np.maximum(1e-10, 1 + alpha * x + beta * np.power(x, 2))

        norm_const = lambda lower_bound, upper_bound, alpha, beta: upper_bound - lower_bound \
            + alpha/2 * (upper_bound ** 2 - lower_bound ** 2) + beta/3 * (upper_bound ** 3 - lower_bound ** 3)
        
        func_norm = lambda x, alpha, beta: 1 / (norm_const(bounds[0], bounds[1], alpha, beta)) * func(x, alpha, beta)
    
        LH_unbinned_object = UnbinnedLH(func_norm, sample_values, bound = bounds, extended = False)
        fit = Minuit(LH_unbinned_object, alpha = alpha_guess, beta = beta_guess)
        fit.errordef = Minuit.LIKELIHOOD
        fit.migrad()

        fig6, ax6 = plt.subplots()
        bins = 50
        binwidth = (bounds[1] - bounds[0]) / bins
        counts, edges, _ = ax6.hist(sample_values, range = bounds, bins = bins, histtype = 'stepfilled', alpha = .4)

        x_vals = 0.5 * (edges[1:] + edges[:-1])
        dy = np.sqrt(counts)
        mask = (counts > 0)

        func_norm_hist = lambda x, alpha, beta: Npoints * binwidth / (norm_const(bounds[0], bounds[1], alpha, beta)) * func(x, alpha, beta)

        chi2_obj = Chi2Regression(func_norm_hist, x_vals[mask], counts[mask], dy[mask])
        chi2_fit = Minuit(chi2_obj, alpha = alpha_guess, beta = beta_guess)
        chi2_fit.migrad()
        x_range = np.linspace(bounds[0], bounds[1], 400)
        fit_vals = func_norm_hist(x_range, *chi2_fit.values[:])
        ax6.plot(x_range, fit_vals)
        Ndof, chi2, p = get_statistics_from_fit(chi2_fit, len(x_vals[mask]))
        print("Chi2 fit vals: ", *fit.values[:])
        print("chi2 ndof chi2 p: ", Ndof, chi2, p)

        if 0:
            fmax = func_norm(bounds[1], *fit.values[:])
            LL_vals, p_val = evaluate_likelihood_fit(func_norm, fmax, fit.values, fit.fval, bounds = bounds, Ndatapoints = Npoints, Nsimulations = 100)
            print("LL fit p-val: ", p_val)
            plot_likelihood_fits(LL_vals, p_val, fit.fval)


        print("minuit est for alpha: ", fit.values['alpha'], "\u00B1", fit.errors['alpha'])
        print("minuit est for beta: ", fit.values['beta'], "\u00B1", fit.errors['beta'])


        alpha_list = []
        alpha_err_list = []
        beta_list = []
        beta_err_list = []

        fig00, ax00 = plt.subplots()
        range = [np.min(sample_values), np.max(sample_values)]
        bins = 50
        binwidth = (range[1] - range[0]) / bins

        for i, dat in enumerate(sample_values_copy):

            ax00.hist(dat, range = range, bins = bins, histtype = 'stepfilled', alpha = .3, lw = 2, label = f'Sample {i}')


            LH_unbinned_object = UnbinnedLH(func_norm, dat, bound = bounds, extended = False)
            fit = Minuit(LH_unbinned_object, alpha = alpha_guess, beta = beta_guess)
            fit.migrad()
            fit.errordef = Minuit.LIKELIHOOD

            x_linvals = np.linspace(bounds[0], bounds[1], 500)
            func_vals = len(dat) * binwidth * func_norm(x_linvals, *fit.values[:])

            ax00.plot(x_linvals, func_vals, label = f'Fit data set {i}')
            
            alpha_list.append(fit.values['alpha'])
            beta_list.append(fit.values['beta'])
                   
            alpha_err_list.append(fit.errors['alpha'])
            beta_err_list.append(fit.errors['beta'])
       
            print("\nminuit est for alpha: ", fit.values['alpha'], "\u00B1", fit.errors['alpha'])
            print("minuit est for beta: ", fit.values['beta'], "\u00B1", fit.errors['beta'])



    
        print("alpha weighted mean, dalpha, Ndof, chi2, p: ", calc_weighted_mean(np.array(alpha_list), np.array(alpha_err_list)))
        print("beta weighted mean, dbeta, Ndof, chi2, p: ", calc_weighted_mean(np.array(beta_list), np.array(beta_err_list)))

        fig00.tight_layout()
        ax00.legend()


        # EXC 3 extra: 





        plt.show()

    if exc3_extra:
        ## Using f = 1 + alpha x + beta x ^ 2 + gamma x ^5 in [-1,1] normalized, gen. 2000 MC points
        # Fit the maximum likelihood estimator
        # Show that MC resampling produces uncertainties sim to DeltaLLH prescription for the 3D hypersurface
        # In 3D, are 500 points enough? Are 2000?
        # Write a profiler to project the 2D (???) contour onto 1D properly

        bounds = [-1,1]

              ## EXC 1: Sim. values acc to pdf and fit estimators each time and plot distribution
        Npoints = 1000
        Nsimulations = 200
        alpha_true, beta_true, gamma_true = 0.5, 0.5, 0.9
        x_linvals = np.linspace(bounds[0], bounds[1], Npoints)

        params_est_list = np.empty([3, Nsimulations])
        params_err_list = np.empty_like(params_est_list)

        def func(x, alpha, beta, gamma):
            return np.maximum(1e-10, 1 + alpha * x + beta * np.power(x, 2) + gamma * np.power(x, 5))

        norm_const = lambda lower_bound, upper_bound, alpha, beta, gamma: upper_bound - lower_bound \
            + alpha / 2 * (upper_bound ** 2 - lower_bound ** 2) + beta / 3 * (upper_bound ** 3 - lower_bound ** 3) \
                + gamma / 6 * (upper_bound ** 6 - lower_bound ** 6)
        
        func_norm = lambda x, alpha, beta, gamma : 1 / (norm_const(bounds[0], bounds[1], alpha, beta, gamma)) * func(x, alpha, beta, gamma)
        fmax = np.max(func_norm(x_linvals, alpha_true, beta_true, gamma_true))
        

        ## Perform MC resampling -- i.e. sim. values acc. to true pdf Nsimulations time and fit estimators each time
        for i in np.arange(Nsimulations):

            sample_values = rejection_sampling_uniform(lambda x: func_norm(x, alpha_true, beta_true, gamma_true),\
                fmax, bounds, Npoints, verbose = False)[0]

            ## Minimize with minuit for reference
            LH_unbinned_object = UnbinnedLH(func_norm, sample_values, bound = bounds, extended = False)
            fit = Minuit(LH_unbinned_object, alpha = alpha_true, beta = beta_true, gamma = gamma_true)
            fit.errordef = Minuit.LIKELIHOOD
            fit.migrad()
            if fit.fmin.is_valid is not True:
                print(f"LL-fit failed for simulation {i}")
            params_est_list[:,i] = fit.values[:]
            params_err_list[:,i] = fit.errors[:]
        
        ## Calc 1sig conf. intervals
         ## Find 1sigma confidence levels. Frac NOT contained in 1sig central interval
        crit_val_lower = stats.norm.cdf(-1, loc = 0, scale = 1)
        crit_val_upper = stats.norm.cdf(1, loc = 0, scale = 1)
        left_index = int(np.ceil(crit_val_lower * Nsimulations ))
        right_index = int(np.ceil(crit_val_upper * Nsimulations ))    
        print("Critical 1sigma indices: ", left_index, right_index)

        # Sort parameter lists
        param_list_sorted = np.sort(params_est_list, axis = 1)

        conf_ints = np.array([param_list_sorted[:, left_index], param_list_sorted[:, right_index]]).T
        print("Conf intervals: ", conf_ints)

        params_arr = np.array([alpha_true, beta_true, gamma_true])
        print("std_param -, std_param + : ", np.abs(conf_ints - params_arr[:, np.newaxis]))


        if 0:
            conf_int_alpha = [alpha_list[left_index], alpha_list[right_index]]
            conf_int_beta = [beta_list[left_index], beta_list[right_index]]
            conf_ints = [conf_int_alpha, conf_int_beta]
        
        ## Plot histograms of fitted values
        fig1, ax1 = plt.subplots(ncols = 3, figsize = (16,9))
        ax1 = ax1.flatten()

        bins = int(Nsimulations/10)
        names = ['Alpha', 'Beta', 'Gamma']
        d = {'Entries': Nsimulations}

        print("minuit avg vals and avg stds: ", np.mean(params_est_list, axis = 1), np.mean(params_err_list, axis = 1))

        for i, params in enumerate(params_est_list):
            range = (np.min(params), np.max(params))
            mean, std = params.mean(), params.std(ddof = 1)
            count, _, _ = ax1[i].hist(params_est_list[i,:], range = range, bins = bins, \
                histtype = 'stepfilled', alpha = .6, lw =2, label = 'Fitted ML values')
            ax1[i].set(xlabel = f'{names[i]}', ylabel = 'count', title = 'simulated ML estimates')
            ax1[i].plot([mean, mean], [0, np.max(count)], label = 'Sample mean')
            ax1[i].plot([conf_ints[i,0], conf_ints[i,0]],  [0, np.max(count)])
            ax1[i].plot([conf_ints[i,1], conf_ints[i,1]],  [0, np.max(count)])

            d0 = {'Mean': mean, 'Std': std}
            d.update(d0)
            text = nice_string_output(d, extra_spacing=2, decimals=3)
            add_text_to_ax(0.05, 0.75, text, ax1[i], fontsize=13)
            ax1[i].legend()
        
        fig1.tight_layout()


        ## Do raster scan of logLH landscape to extract param uncertainties

        # Step 1: Calc delta lnL val for 3 params
        sig1 = 1 - 2 * stats.norm.sf(1,0,1)

        ## -2LLH is chi2 dist with dof = N_params. We need to solve sig1 = stats.chi2.cdf(x,Ndof) to find delta(2LLH) = x corresponding to 1 sig conf int.
        chi2_vals = np.arange(3,4.5,0.01)
        chi2_cdf_vals = stats.chi2.cdf(chi2_vals,len(params_arr))
        crit_val_ind = np.argmin(np.abs(chi2_cdf_vals - sig1))
        delta_LLH = 0.5 * chi2_vals[crit_val_ind]
        #delta_LLH = [0.5]


        N_scan_points = 50
        alpha_width, beta_width, gamma_width = 0.3, 0.4, 0.6
        alpha_est, beta_est, gamma_est = fit.values['alpha'], fit.values['beta'], fit.values['gamma']
        alpha_err, beta_err, gamma_err = fit.errors['alpha'], fit.errors['beta'], fit.errors['gamma']
        
        alpha_vals = np.linspace(alpha_est - alpha_width, alpha_est + alpha_width, N_scan_points)
        beta_vals = np.linspace(beta_est - beta_width, beta_est + beta_width, N_scan_points)
        gamma_vals = np.linspace(gamma_est - gamma_width, gamma_est + gamma_width, N_scan_points)

        ALPHA, BETA, GAMMA = np.meshgrid(alpha_vals, beta_vals, gamma_vals)

        logLH_max = - fit.fval
        delta_logLH_val_3_params = np.array([delta_LLH])

        print("\alpha beta gamma est and logLH: ", alpha_est, beta_est, gamma_est, logLH_max)
        print("alpha beta gamma minuit std: ", alpha_err, beta_err, gamma_err)

        func_norm_vec = lambda x: 1 / (norm_const(bounds[0], bounds[1], ALPHA, BETA, GAMMA)) * func(x, ALPHA, BETA, GAMMA)

        ## Calc logLH landscape

        ##NBNBNBNBNBNB: logLH takes three entries having the format (beta,alpha,gamma), ie beta is the first entry corres. to matrix number. alpha varies over rows and gamma
        ## over columns
        logLH = np.empty_like(ALPHA)

        for val in sample_values:
            logLH += np.log(func_norm_vec(val))


        print("Raster logLH max: ", np.max(logLH))

        ind = np.argwhere(np.abs(logLH - logLH_max) < delta_logLH_val_3_params)
      
        betas_within_1sig = beta_vals[ind[:,0]]
        alphas_within_1sig = alpha_vals[ind[:,1]]
        gammas_within_1sig = gamma_vals[ind[:,2]]

        alpha_conf_int = np.array([np.min(alphas_within_1sig), np.max(alphas_within_1sig)])
        beta_conf_int = np.array([np.min(betas_within_1sig), np.max(betas_within_1sig)])
        gamma_conf_int = np.array([np.min(gammas_within_1sig), np.max(gammas_within_1sig)])

        print("Raster scan conf intervals for alpha beta gamma :", alpha_conf_int, beta_conf_int, gamma_conf_int)
        print("alpha std_param -, std_param + : ", np.abs(alpha_conf_int - alpha_est))
        print("beta std_param -, std_param + : ", np.abs(beta_conf_int - beta_est))
        print("gamma std_param -, std_param + : ", np.abs(gamma_conf_int - gamma_est))

        if 0:
            fig0, ax0 = plt.subplots()
            im1 = ax0.imshow(logLH, extent = (alpha_vals[0], alpha_vals[-1], beta_vals[0], beta_vals[-1]), origin= 'upper')

            #plot contours and extraxt contour points
            Ncontours = 3
            alpha_contour_vals = [None] * Ncontours
            beta_contour_vals = [None] * Ncontours

            for i, LH_val in enumerate(delta_logLH_vals_2_params):
                im = ax0.contour(ALPHA, BETA, logLH, levels = [logLH_max - delta_logLH_vals_2_params[i]], cmap = 'viridis', alpha = .6) #, 
                v = im.collections[0].get_paths()[0].vertices
                alpha_contour_vals[i] = v[:,0]
                beta_contour_vals[i] = v[:,1]

            ## Now extract min and max values of alpha and beta to obtain 1sig conf interval
            alpha_1sig_int = [np.min(alpha_contour_vals[0]), np.max(alpha_contour_vals[0])]
            beta_1sig_int = [np.min(beta_contour_vals[0]), np.max(beta_contour_vals[0])]
            print("Raster scan conf int for alpha, beta: ", alpha_1sig_int, beta_1sig_int)
            print("Raster std alpha + and std alpha -: ", alpha_1sig_int[1] - alpha_est, alpha_est - alpha_1sig_int[0])
            print("Raster std beta + and std beta -: ", beta_1sig_int[1] - beta_est, beta_est - beta_1sig_int[0])


            if 0:
                divider = make_axes_locatable(ax0)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig0.colorbar(im, cax=cax, orientation='vertical')

            if 1:
                divider = make_axes_locatable(ax0)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig0.colorbar(im1, cax=cax, orientation='vertical')

            ax0.plot(alpha_est, beta_est, 'x', lw = 2.5, markersize = 12)
            ax0.set(xlabel = r'$\alpha$', ylabel = r'$\beta$', title = r'$\ln LH$')







        plt.show()



if __name__ == '__main__':
    main()
