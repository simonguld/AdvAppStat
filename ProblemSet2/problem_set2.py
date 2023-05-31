# Author: Simon Guldager Andersen
# Date (latest update): 28-02-2023

### SETUP ------------------------------------------------------------------------------------

## Imports:
import os, sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats, interpolate, optimize
from iminuit import Minuit
from matplotlib import rcParams
from cycler import cycler

sys.path.append('Appstat2022\\External_Functions')
from ExternalFunctions import Chi2Regression, BinnedLH, UnbinnedLH
from ExternalFunctions import nice_string_output, add_text_to_ax    # Useful functions to print fit results on figure


## Change directory to current one
os.chdir('AdvAppStat\ProblemSet2')


### FUNCTIONS ----------------------------------------------------------------------------------

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
    N_try = int(2 * Npoints)
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

        fig0, ax0 = plt.subplots()

        ax0.grid(False)
    
        counts, edges, _ = plt.hist(LL_values, bins = 50, histtype = 'step', lw = 2, color = 'red');
        x_vals = 0.5 * (edges[:-1] + edges[1:])
        ax0.plot([log_likelihood_val, log_likelihood_val], [0,np.max(counts)], 'k--', label = ' - Log likelihood value (from fit)', lw = 2)
        ax0.set(xlabel = r'$-\ln$LH', ylabel = 'Count', title = r'Simulated $-\ln$LH values', ylim = (0,np.max(counts) * 1.1))

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
        fig0.legend( fontsize = 14, bbox_to_anchor = (0.25,0.65,0.25,0.25))
        fig0.tight_layout()
        return None

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

def P0():
    # Problem 0: Generate a plot of Gaussian distribution with 
    mu, sigma = 10, np.sqrt(2.3)

    x_range = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 1000)
    vals = stats.norm.pdf(x_range, loc = mu, scale = sigma)

    fig, ax = plt.subplots()
 
    ax.plot(x_range, vals, label = r'$\mathcal{N}(\mu = 10, \sigma^2 = 2.3)$')
    ax.legend(fontsize = 18)
    plt.show()

def P1():
    # 1) Gen 807 values in range [-1.02, 1.11] of pdf f = 1 + alpha x + beta x ^2 for alpha = 0.9 and beta = .55
    # 2) Gen 513 values using a pmf propto poisson dist with lambda = 3.8 and x in [0,infty]
    ## In both cases:
    # Find MLE in each case
    # Histogram data and plot the fitted function 
    # explain how you ensure normalization
    # save data sets to two txt files

    ### Case 1:
     ## EXC 1: Sim. values acc to pdf and fit estimators each time and plot distribution

        parabola_fit, poisson_fit = True, True

        if parabola_fit:
            Npoints1 = 807
            bounds = [- 1.02, 1.11]
            alpha_true, beta_true = 0.9, 0.55

            # Define function and normalize
            def func(x, alpha, beta):
                return np.maximum(1e-14, 1 + alpha * x + beta * np.power(x, 2))

            norm_const = lambda lower_bound, upper_bound, alpha, beta: upper_bound - lower_bound \
                + alpha/2 * (upper_bound ** 2 - lower_bound ** 2) + beta/3 * (upper_bound ** 3 - lower_bound ** 3)
            
            func_norm = lambda x, alpha, beta: 1 / (norm_const(bounds[0], bounds[1], alpha, beta)) * func(x, alpha, beta)
    
            # Calc fmax
            fmax = max(func_norm(bounds[0], alpha_true, beta_true), func_norm(bounds[1], alpha_true, beta_true), -alpha_true / (2 * beta_true))

            # Generate sample values
            sample_values = rejection_sampling_uniform(lambda x: func_norm(x, alpha_true, beta_true),\
            fmax, bounds, Npoints1, verbose = True)[0]
    
            ## Save sample values
            #np.savetxt('parabola_samples.txt', sample_values)


            ## Minimize with minuit
            LH_unbinned_object = UnbinnedLH(func_norm, sample_values, bound = bounds, extended = False)
            fit = Minuit(LH_unbinned_object, alpha = alpha_true, beta = beta_true)
            fit.errordef = Minuit.LIKELIHOOD
            fit.migrad()
            if fit.fmin.is_valid is not True:
                print(f"LL-fit failed")
            print("MLE of alpha: ", [fit.values['alpha'],fit.errors['alpha']])
            print("MLE of beta: ", [fit.values['beta'],fit.errors['beta']])
        

            ## Evaluate goodness of likelihood fit
            LL_vals, p_val = evaluate_likelihood_fit(func_norm, fmax, fit.values[:], fit.fval, bounds = bounds, Ndatapoints = Npoints1, Nsimulations = 1000)
            plot_likelihood_fits(LL_vals, p_val, fit.fval)

            # Plot histogram of data along with fit
            fig1, ax1 = plt.subplots()
            bins = 40
            binwidth = (bounds[1] - bounds[0]) / bins
        
            ax1.set(xlabel = 'x-value', ylabel = 'count', title = 'Simulated values along with ML-fit')
            count, edges, _ = ax1.hist(sample_values, range = bounds, bins = bins, histtype = 'stepfilled', alpha = .3, lw = 2, label = 'Simulated values')

    
            # Calc and plot fit values
            x_vals = np.linspace(bounds[0], bounds[1], 1000)
            func_norm_scaled = lambda x, alpha, beta: Npoints1 * binwidth * func_norm(x, alpha, beta)
            f_vals = func_norm_scaled(x_vals, *fit.values[:])
            ax1.plot(x_vals, f_vals, label = '(Unbinned) maximum likelihood fit')
        
            ## Verify with chi2 fit
            bin_centers = 0.5 * (edges[1:] + edges[:-1])
            dy = np.sqrt(count)
            func_norm_ext = lambda x, alpha, beta, N: N * binwidth * func_norm(x, alpha, beta)
            fitchi2 = do_chi2_fit(func_norm_ext, bin_centers, count, dy, np.array([fit.values['alpha'], fit.values['beta'], Npoints1]))
            Ndof, chi2, p = get_statistics_from_fit(fitchi2, len(bin_centers), subtract_1dof_for_binning = False)
            print(chi2/Ndof, p)

            d = {'Entries': Npoints1, 'alpha': [fit.values['alpha'],fit.errors['alpha']], 'beta': [fit.values['beta'],fit.errors['beta']]}
            text = nice_string_output(d, extra_spacing=2, decimals=2)
            add_text_to_ax(0.05, 0.75, text, ax1, fontsize=15)
    
            ax1.legend(loc = 'upper left')
            fig1.tight_layout()

        if poisson_fit:
            Npoints2 = 513
            mean_true = 3.8
            end_scale = 4
            bounds = [0.0, np.ceil(end_scale * mean_true)]
            print("Using the truncated interval ", bounds, " results in the loss of ", \
                stats.poisson.sf(bounds[1], mu = mean_true) * 100, " percent of all points")

            # Already normalized, in the discrete sense
            pmf = lambda k, mean: stats.poisson.pmf(k, mu = mean)
         
            # Gen Npoints2 values according to pmf
            sample_values = stats.poisson.rvs(mu = mean_true, size = Npoints2)
          
            while np.max(sample_values) > bounds[1]:
                print("Generated Poisson values exceeded range. They will be resimulated until all values are contained within range." )
                sample_values = stats.poisson.rvs(mu = mean_true, size = Npoints2)
            # Save values
            #np.savetxt('andersen_poisson_samples.txt', sample_values)

            ## Minimize by hand:
            logLH = lambda mu: - np.sum(np.log(pmf(sample_values, mu)))
            logLH_vec = np.vectorize(logLH)
            mu = np.linspace(3,4,10000)
            logLH_vals = logLH_vec(mu)
            logLH_min = np.min(logLH_vec(mu))
            mu_opt = mu[np.argmin(logLH_vec(mu))]
            print("Scan -logLH and lambda: ", logLH_min, mu_opt)

            ## Extract 1sig conf levels
            critical_value = 0.5
            indices_within_1sig = np.argwhere(np.abs(logLH_vals - logLH_min) < critical_value)
            mu_within_1sig = np.sort(mu[indices_within_1sig])
            std_mu_lower, std_mu_upper = mu_within_1sig[0], mu_within_1sig[-1]
            std_mu_av = 0.5 * (std_mu_upper - std_mu_lower)

            print("lambda 1 sigma confidence level: ",std_mu_lower, std_mu_upper)
            print("lower and upper 1 sigma uncertainty: ", np.abs(mu_opt - mu_within_1sig[0]), np.abs(mu_opt - mu_within_1sig[1]))
            print("average uncertainty on lambda: ", 0.5 * (std_mu_upper - std_mu_lower))

            res = optimize.minimize(logLH, x0 = 3.5)
            if res.success:
                print("\nScipy minimization was a succes")
            print("Scipy lambda -lnLH: ", res.x, "  ", logLH(res.x))


            ## Plot histogram of data along with fit
            fig2, ax2 = plt.subplots()
            bins = int(bounds[1] - bounds[0] )
            binwidth = 1
        
            ax2.set(xlabel = '$k$', ylabel = 'count', title = 'Simulated values along with ML-fit')
            count, edges, _ = ax2.hist(sample_values, range = (bounds[0]-0.5,bounds[1] - 0.5), bins = bins, histtype = 'stepfilled', alpha = .3, lw = 2, label = 'Simulated values')

    
            # Calc and plot fit values
            x_vals = np.arange(bounds[1]+1)
            pmf_scaled = lambda k, mean: Npoints2 * binwidth * pmf(k, mean)

            f_vals = pmf_scaled(x_vals, res.x)
            ax2.plot(x_vals, f_vals, '.-', label = '(Unbinned) maximum likelihood fit')
        
         
            ## Verify with chi2 fit
            bin_centers = edges[:-1] + 0.5
            dy = np.sqrt(count)
            mask = (count > 0)
         
            chi2 = np.sum((pmf_scaled(bin_centers[mask], res.x) - count[mask]) ** 2 / dy[mask] ** 2)
            Ndof = len(bin_centers[mask]) - 1 - len(res.x) # -1 cuz of normalization
            p_val = stats.chi2.sf(chi2, Ndof)
            

            ax2.errorbar(bin_centers, count, dy, fmt = 'k.', elinewidth = 1, capthick = 1, capsize = 1)
            d = {'Entries': Npoints2, 'Lambda': [float(res.x),float(std_mu_av)], 'Ndof': Ndof, 'Chi2': chi2, 'p-val': p_val}
            text = nice_string_output(d, extra_spacing=2, decimals=2)
            add_text_to_ax(0.55, 0.75, text, ax2, fontsize=15)
    
            ax2.legend(loc = 'upper right')
            fig2.tight_layout()



        plt.show()

def P2():
    ## Download outline given by x,y data points connected through lin. interpolation that outline a contained area
    ## Using MC sampling, estimate the area and visualize
    x_vals, y_vals = np.loadtxt('OutlineAreaSpline.txt', skiprows = 1, unpack = True)
    Npoints = len(x_vals)

    ## Visualize outline
    fig1, ax1 = plt.subplots()

    ax1.plot(x_vals, y_vals, 'r.')
    ax1.set(xlabel = '$x$', ylabel = '$y$', title = 'Region contained by outline along with simulated points')


    ## Find min and max of x and divide into 2 subsets
    xmin_ind, xmax_ind = np.argmin(x_vals), np.argmax(x_vals)

    ## Define upper and lower valus needed to construct 2 spline functions
    index = np.arange(Npoints)
    mask = ((index >= xmin_ind) & (index <= xmax_ind))
    mask_lower = ((~mask) | (index == xmin_ind) | (index == xmax_ind))
    x_upper, y_upper = x_vals[mask], y_vals[mask]

    ## Sort lower values
    sort_ind_lower = np.argsort(x_vals[mask_lower])
    x_lower, y_lower = np.sort(x_vals[mask_lower]),y_vals[mask_lower][sort_ind_lower]

    # Construct upper and lower spline functions
    f_spline_upper = interpolate.interp1d(x_upper, y_upper, kind = 'linear')
    f_spline_lower = interpolate.interp1d(x_lower, y_lower, kind = 'linear')


    # NEXT STEPS:
    # Define domain. Simulate points. Count N contained in region. Est area and uncertainty if possible
    # Plot (some) of the simulated points on ax1 with appropriate labels
    x_range = [np.min(x_vals), np.max(x_vals)]
    y_range = [np.min(y_vals), np.max(y_vals)]
    ranges = np.r_['0,2', x_range, y_range]


    # Define no. of subdivisions
    Nsubdivisions = 100_000

    # Simulate values
    sample_values = ranges[:,0][:,np.newaxis] \
        + stats.uniform.rvs(size = (2, Nsubdivisions)) * (ranges[:,1][:,np.newaxis] - ranges[:,0][:,np.newaxis])

    # Make a mask to see which bounds are within region
    mask_within = ((f_spline_upper(sample_values[0,:]) >= sample_values[1,:]) \
        & (f_spline_lower(sample_values[0,:]) <= sample_values[1, :]))

    values_within = sample_values.T[mask_within].T
    values_outside = sample_values.T[~mask_within].T

    # Calc fractions inside and outside and from that the area within
    fraction_within = len(values_within[0, :]) / Nsubdivisions
    fraction_within_err = np.sqrt(fraction_within * (1 - fraction_within) / Nsubdivisions)

    area_square = (x_range[1] - x_range[0]) * (y_range[1] - y_range[0])
    area_within = fraction_within * area_square
    ## Estimate uncertainty as binomial fractional uncertainty ('counting uncertainty')
    area_within_err = fraction_within_err * area_square

    print(area_square, fraction_within)
    print("Area bounded by outline: ", area_within, "\u00B1", area_within_err)

    # Plot simulated points
    ax1.plot(values_within[0,:], values_within[1,:], 'b.', markersize = 4)
    ax1.plot(values_outside[0,:], values_outside[1,:],'.', color = 'purple', markersize = 4)
    ## Plot spline functions
    ax1.plot(x_lower, f_spline_lower(x_lower), color = 'black', label = 'Lower lin. spline function')
    ax1.plot(x_upper, f_spline_upper(x_upper), color = 'coral', label = 'Upper lin. spline function')
    
    d = {'Entries': Nsubdivisions, 'Area': [area_within, area_within_err]}

    text = nice_string_output(d, extra_spacing=2, decimals=3)
    add_text_to_ax(0.57, 0.22, text, ax1, fontsize=15)
    


    ax1.legend(loc = 'lower left')
    fig1.tight_layout()
    plt.show()

def P4():
    ## Estimate the mean and the interquartile range of a lake fish population satisfying:
    # Lake volume estimate is Gaussian with 5000 \pm 300 m^3. volume/fish estimate is Gaussian with 10 \pm 1 m^3
    # The total no of fish, V_lake / V_fish is a fraction of Gaussians, which is not itself Gaussian, so do not assume symmetric uncertainties 

    mu_lake, std_lake = 5000, 300
    mu_fish, std_fish = 10, 1

    fish_distribution = lambda size: stats.norm.rvs(loc = mu_lake, scale =std_lake, size = size) / stats.norm.rvs(loc = mu_fish, scale = std_fish, size = size)
   
    ## Simulate values according to the fish distrubtion and histogram the data
    Nsamples = 1_000_000
    sample_values = np.sort(fish_distribution(size = Nsamples))



    ## Histogram the data    
    range = (300, 800)
    bins = int(Nsamples / 1000)

    fig,ax = plt.subplots()
    ax.set(xlabel = 'Number of fish', ylabel = 'Count', title = 'Simulated values of the number of fish')
    count, edges, _ = ax.hist(sample_values, range = range, bins = bins, histtype = 'stepfilled', alpha = .3)
    
    ## Calc mean and interquartile range
    sample_mean = sample_values.mean()
    sample_std = sample_values.std(ddof = 1)
    sample_sem = sample_std / np.sqrt(Nsamples)

    # Define critical indices
    confidence_lower, confidence_upper = .25, .75
    index_lower, index_upper = int(np.ceil(confidence_lower * Nsamples)), int(np.ceil(confidence_upper * Nsamples))

    sample_interquartile_range = [sample_values[index_lower] , sample_values[index_upper]]
 
    bin_centers = 0.5 * (edges[1:] + edges[:-1])
    bin_ind_lower = np.argmin(np.abs(bin_centers - sample_interquartile_range[0]))
    bin_ind_upper = np.argmin(np.abs(bin_centers - sample_interquartile_range[1]))

    ## Plot mean and interquartile range
    ax.plot([sample_mean, sample_mean], [0, np.max(count)], label = 'Sample mean')
    ax.plot([sample_interquartile_range[0], sample_interquartile_range[0]], [0,count[bin_ind_lower]], label = '1st quartile')
    ax.plot([sample_interquartile_range[1], sample_interquartile_range[1]], [0,count[bin_ind_upper]], label = '3rd quartile')

    ## Add data to plot
    d = {'Entries': Nsamples, 'Sample mean': [sample_mean,sample_sem], '1st quartile': sample_interquartile_range[0], '3rd quartile': sample_interquartile_range[1]}
    text = nice_string_output(d, extra_spacing=1, decimals=2)
    add_text_to_ax(0.53, 0.7, text, ax, fontsize=15)

    ax.legend()
    plt.show()


def main():
    
    ## Set which problems to run
    p0, p1, p2,  p4 = True, True, True,  True
    problem_numbers = [p0, p1, p2, p4]
    f_list = [P0, P1, P2, P4]

    for i, f in enumerate(f_list):
        if problem_numbers[i]:
            print(f'\nPROBLEM {i}:')
            f()


if __name__ == '__main__':
    main()
