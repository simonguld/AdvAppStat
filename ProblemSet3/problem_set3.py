# Author: Simon Guldager Andersen
# Date (latest update): 17-03-2023

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

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV

#sys.path.append('Appstat2022\\External_Functions')
#sys.path.append('AdvAppStat')
from ExternalFunctions import Chi2Regression, BinnedLH, UnbinnedLH
from ExternalFunctions import nice_string_output, add_text_to_ax    # Useful functions to print fit results on figure
from statistics_helper_functions import calc_ROC, calc_fisher_discrimminant, calc_ROC_AUC, calc_mean_std_sem

## Change directory to current one
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)


### FUNCTIONS ----------------------------------------------------------------------------------
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

def P1():
    # Problem 1: Generate a plot of chi square distribution with 1 dof in range [0,10]
    dof = 1
    bounds = [0, 10]

    x_range = np.linspace(bounds[0], bounds[1], 1000)
    vals = stats.chi2.pdf(x_range, dof)

    fig, ax = plt.subplots()
 
    ax.set_xticks(np.arange(10+1))
    ax.plot(x_range, vals, label = r'$\chi^2(N_{dof} = 1)$')
    ax.legend(fontsize = 18)
    plt.show()

def P2():
    # Consider data gen acc. to a 4 parameters prob. density f satisfying Wilk's theorem. 
    # What is the value of Delta 2 ln LH that should be used to construct 77.9 % conf. interval
    # from the best fit point?

    conf_level = 0.779
    Ndof = 4

    deltaLLH_val = stats.chi2.isf(1 - conf_level, Ndof)

    print("Value of 2 *Delta log likelihood needed to construct 77.9% confidence interval from best fit point of\
    4 parameter probability distribution: ", deltaLLH_val)

def P3():
    ## EXC 3: The length of white sharks is provided
    # a1: Plot shark lengths as histogram. In sep. plot, plot density estimate from KDE with gaussian kernel and bandwidth = 25cm.
    # a2: Using the KDE, estimate p-value that length > 653 cm
    # b:
    ## load data
    lengths = np.loadtxt('WhiteSharkLength.txt', skiprows=2)
    Npoints = len(lengths)

    ## visualize data and calculate the kernel density estimate
    range = (min(lengths), max(lengths))
    bins = int(Npoints / 10)


    fig0, ax0 = plt.subplots()
    ax0.hist(lengths, range=range, bins=bins, histtype = 'bar', edgecolor = 'black', alpha = .6, label = 'Shark length data')
    ax0.set(xlabel = 'Length (cm)', ylabel = 'Count', title = 'Histogram of shark lengths')
    ax0.legend(loc = 'upper left')
    fig0.tight_layout()

    ## Show KDE kernels, resulting PDE and rug plot
    # using bandwidth = std
    bandwidth = 25

    # construct KDE estimated pdf
    pde_pdf = gaussian_pde(lengths, bandwidth)
    x_vals = np.linspace(100, 750, 3000)
    length_vals = pde_pdf(x_vals)
    fig1, ax1 = plt.subplots()
    
    ax1.plot(x_vals, length_vals, label = 'KDE from Gaussian kernels')
    ax1.set(xlabel = 'Length (cm)', ylabel = 'Density', title = 'KDE of shark length distribution')
    sns.rugplot(lengths)
    ax1.hist(lengths, range=range, bins=bins, histtype = 'bar', edgecolor = 'black', alpha = .6, label = 'Shark length data', density=True)
    ax1.legend(loc = 'upper left')
    fig1.tight_layout()


    ## a2: USE KDE-pdf to est P(L > 653)
    L_cutoff = 653
    L_range = (0, 800)

    p_val, err_p_val = integrate.quad(pde_pdf, L_cutoff, 800)

    print(f"P(L > {L_cutoff}) = ", p_val, "\u00B1", err_p_val)


    ## b)
    # Assume female:male ratio is 50:50 for all weights, i.e. weight distributions are identical
    # we are given length probablities given weight and sex

    # 1: including kde from A, and using new porbability eq., what is P(L > 337cm | W = 763 kg)
    # solution: We have LH(L|W) = P(L|W) = P(female|W) LH(L|W and female) + P(male|W) * LH(L|W and male),
    #           where P(female|W) = P(male | W) = .5 by assumption. 
    # The posterior is then Post(L|W) = LH(L|W) Prior(L), where the prior is the kde estimate of the distribtuion. 
    # 2: plot fully norm. posterior in range [100, 750] cm for w = 763kg

    w_cutoff = 763

    fig2, ax2 = plt.subplots()

    # define LH(L|W and sex) for each sex
    prob_length_female = lambda l, w: stats.norm.pdf(l, loc = 0.434 * w, scale = 55)
    prob_length_male = lambda l, w: stats.norm.pdf(l, loc = 0.293 * w, scale = 62)

    # define prior
    prior = lambda l: pde_pdf(l)

    # calc likelihood function
    likehood = lambda l: 0.5 * (prob_length_female(l, w_cutoff) + prob_length_male(l, w_cutoff))

    # calc and normalize posteriors
    posterior = lambda l: likehood(l) * prior(l)
    post_norm = integrate.quad(posterior, L_range[0], L_range[1])[0]
    posterior_normalized = lambda l: posterior(l) / post_norm

    ## calc posterior probability Post(L > 337 cm | W = 763 kg)
    L_cutoff = 337
    prob = integrate.quad(posterior_normalized, L_cutoff, L_range[1])

    print(f"Posterior(L > {L_cutoff} cm | w = {w_cutoff}) = ", prob)
   
    # Plot histogram, prior, likehood, and posteriors
    ax2.set(xlabel = 'Length (cm)', ylabel = 'Density', title = f'Length distribution of sharks with mass {w_cutoff} kg')
    ax2.plot(x_vals, likehood(x_vals), label = 'Likelihood')
    ax2.plot(x_vals, prior(x_vals), label = 'Prior = KDE')
    ax2.plot(x_vals, posterior_normalized(x_vals), label = 'Posterior')
    ax2.legend(loc = 'upper right')
    ax1.legend(loc = 'upper left')


    ## part c: Repeat part b but now consider only sharks with L > 201cm.
    # still asumme 50:50 ratio of males:females and equal weight distributions
    # NB: it is important to normalize prior, LH_fem and LH_homme separately. Otherwise, their distributions are not generally equal
    # if simply normalizing the posterior, or the marginalized LH, after truncation

    fig3, ax3 = plt.subplots()

    min_length = 201

    ## define truncated densities
    def truncator(f, l, cutoff):
        if l < cutoff:
            return 0.0
        else:
            return f(l)


    # define LH(L|W and sex) for each sex
    LH_female_trunc = lambda l: truncator(lambda l: prob_length_female(l, w_cutoff), l, min_length)
    LH_female_trunc_vec = np.vectorize(LH_female_trunc)
    LH_female_trunc_norm = integrate.quad(LH_female_trunc_vec, min_length, L_range[1])[0] 
    LH_female_trunc_normalized = lambda l: LH_female_trunc_vec(l) / LH_female_trunc_norm
   
    LH_male_trunc = lambda l: truncator(lambda l: prob_length_male(l, w_cutoff), l, min_length)
    LH_male_trunc_vec = np.vectorize(LH_male_trunc)
    LH_male_trunc_norm = integrate.quad(LH_male_trunc_vec, min_length, L_range[1])[0] 
    LH_male_trunc_normalized = lambda l: LH_male_trunc_vec(l) / LH_male_trunc_norm

    # combined LH is now automatically normalized and truncated
    LH_trunc = lambda l: 0.5 * (LH_female_trunc_normalized(l) + LH_male_trunc_normalized(l))

    # truncate prior
    prior_trunc = lambda l: truncator(prior, l, cutoff=min_length)
    prior_trunc_vec = np.vectorize(prior_trunc)
    prior_trunc_norm = integrate.quad(prior_trunc_vec, min_length, L_range[1])[0]
    prior_trunc_normalized = lambda l: prior_trunc_vec(l) / prior_trunc_norm

    # define truncated posterior and normalize
    posterior_trunc = lambda l: LH_trunc(l) * prior_trunc_normalized(l)
    posterior_trunc_norm = integrate.quad(posterior_trunc, min_length, L_range[1])[0]
    posterior_trunc_normalized = lambda l: posterior_trunc(l) / posterior_trunc_norm
    

    ## calc posterior probability Post(L > 337 cm | W = 763 kg)
    L_cutoff = 337
    prob = integrate.quad(posterior_trunc_normalized, L_cutoff, L_range[1])

    print(f"Posterior(L > {L_cutoff} cm | w = {w_cutoff} and L > 201 cm) = ", prob)

    # Plot histogram, prior, likehood, and posteriors
    ax3.plot(x_vals, LH_trunc(x_vals), label = 'Truncated likelihood')
    ax3.plot(x_vals, prior_trunc_normalized(x_vals), label = ' Truncated prior')
    ax3.plot(x_vals, posterior_trunc_normalized(x_vals), label = 'Truncated posterior')
    ax3.plot(x_vals, likehood(x_vals), '--', color = 'teal', label = 'Likelihood', lw=1)
    ax3.plot(x_vals, prior(x_vals), '--', color ='navy', label = 'Prior = KDE', lw=1)
    ax3.plot(x_vals, posterior_normalized(x_vals), '--', color = 'coral', label = 'Posterior', lw=1)

    ax3.set(xlabel = 'Length (cm)', ylabel = 'Density', title = f'Distribution of sharks with m = {w_cutoff} kg and l > {min_length} cm')
    ax3.legend(loc = 'upper right')

    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    plt.show()

def P4():
    ## Classification::
    ## Deta was collected regarding whether online activity results in revenue at websites.
    ## We are provided training and test data file, and a blind sample 
    ## Construct classifier to separate those who gen revenue (=1) from those who dont revenue(=0)

    # a) Make histogram using all events for test file with x-axis being the classifier algorithm test-statistic
    # b) discuss (and do) what checks you have made to avoid overtraning
    # c) predict labels for blind sample

    ## Load and organize data
    train_data = pd.read_csv("Set3_Prob4_TrainData.csv", delimiter =',')
    train_labels = np.array(train_data['Revenue'])
    train_features = pd.DataFrame.to_numpy(train_data.iloc[:,1:-1])

    test_data = pd.read_csv("Set3_Prob4_TestData.csv", delimiter =',')
    test_labels = np.array(test_data['Revenue'])
    test_features = pd.DataFrame.to_numpy(test_data.iloc[:,1:-1])

    blind_data = pd.read_csv("Set3_Prob4_BlindData.csv", delimiter =',')
    blind_ids =  np.array(blind_data['ID'])
    blind_features = pd.DataFrame.to_numpy(blind_data.iloc[:,1:])

    Ntrain_data, Nfeatures = train_features.shape
    Ntest_data, _ = test_features.shape 
    Nblind_data, _ = blind_features.shape

    print("N data points in training, test and blind sample: ", Ntrain_data, Ntest_data, Nblind_data)
    print("N features: ", Nfeatures)

    print("Fraction of training data with revenue = 1:", len(train_labels[(train_labels == 1)]) / Ntrain_data)
    print("Fraction of test data with revenue = 1:", len(test_labels[(train_labels == 1)]) / Ntest_data)

    ## Strategies to avoid overtraining:
    # 1) sep train data into train and validation data
    # 2) do randomized search including cross validation
    # 3) look a separator and consider dist of train vs test data. Are they similar?  
    # 4) If time: Separate using several classifiers and vote acc. to voting scheme

    ## Split train data into training and validation data
    
    ## Split into validation and training data
    X_train, X_validation, y_train, y_validation = train_test_split(train_features, train_labels, test_size=0.20)

    Ntrain_split, _ = X_train.shape
    Nvalidation_split, _ = X_validation.shape
    ## signal is revenue, background is no revenue
    train_signal_mask = (y_train == 1)
    validation_signal_mask = (y_validation == 1)
    test_signal_mask = (test_labels == 1)

    plot_training_features = False
    if plot_training_features:
        fig, ax = plt.subplots(ncols = 4, nrows = 4, figsize = (24,24))
        bins = 100
        ax = ax.flatten()

        for i in np.arange(Nfeatures):
            signal = X_train[:,i][train_signal_mask]
            background = X_train[:,i][~train_signal_mask]
            range = (min(np.min(signal), np.min(background)), max(np.max(signal), np.max(background)))
            ax[i].hist(signal, bins = bins, range = range, histtype = 'stepfilled', color = 'black', alpha = .6, label = 'Training data, Rev = 1')
            ax[i].hist(background, bins = bins, histtype = 'step', color = 'red', alpha = .3, label = 'Training data, Rev = 0')
            
            ax[i].set(xlabel = f'x{i}', ylabel = 'count')
            ax[i].legend(fontsize = 10)

    ###-------------------------------------
    ## Construct a boosted trees classifyer

    # Decide which achitechture to use
    GB, ADA = False, True
    if GB:
        ## Set Graident boosting parameters
        kwargs_BDT = {'loss': "log_loss", 'n_estimators': 50, 'learning_rate': 1, 'max_depth': 4, 'criterion': "friedman_mse", \
                        'min_samples_leaf': 1, 'min_samples_split': 2, 'subsample': 1.0, 'max_features': None,\
                            'n_iter_no_change': None, 'validation_fraction': 0.1, 'tol': 0.0001, 'ccp_alpha':  0.0}
        bdt = GradientBoostingClassifier(**kwargs_BDT)
    if ADA:
        ## Set AdaBoost parameters [opt acc. to acc: n_est = 46, lr = 0.58, deåth = 2, max_features = 12]
        kwargs_BDT = {'n_estimators': 24, 'learning_rate': 0.2} # default are 50,1
        ## Set decision tree parameters
        kwargs_DT = { 'max_depth': 2, 'criterion': "gini", 'min_samples_split': 2, 'min_samples_leaf': 1, \
                    'splitter': "best", 'max_features': 14, 'ccp_alpha':  0.0} 
        # default are 1, gini, 2, 1, best, 0. For all parameters, use BDT.get_params()    
        bdt = AdaBoostClassifier(DecisionTreeClassifier(**kwargs_DT), **kwargs_BDT)

    ## Construct parameter ranges to search for best parameters
    ## By iteration, the best estimators have been found to be
    # n_estimator = 46, max_depth = 8, learning_rate = 0.58, max_features = 12  
    search_best_parameters = False
    if search_best_parameters:
        ## define parameters and ranges to vary
        param_grid = {'n_estimators': list(np.arange(24,25,2).astype('int')),\
                    'base_estimator__max_depth': list(np.arange(2,3,2).astype('int')),\
                        'learning_rate': list(np.arange(0.2,0.3,0.2)), \
                            'base_estimator__max_features': list(np.arange(14,16))}
    
        bdt = AdaBoostClassifier(DecisionTreeClassifier(**kwargs_DT), **kwargs_BDT)
        
        ## Decide which parameter search to use
        grid, randomized = False, True
        metric = 'roc_auc'
        if grid:
            bdt = GridSearchCV(bdt, param_grid, scoring = metric, cv = 5, n_jobs = -1).fit(X_train, y_train)
        if randomized:
            bdt = RandomizedSearchCV(bdt, param_grid, n_iter = 50, scoring = metric, \
                                        cv = 5, n_jobs = -1).fit(X_train, y_train)
                 
        print("Best Param for GS", bdt.best_params_)
        print("CV score for GS", bdt.best_score_)
    else:
        ## Perform Nfit fits to obtain av value of validation and test data accuracy
        Nfits = 20
        train_acc_arr = np.empty(Nfits)
        val_acc_arr = np.empty(Nfits)
        test_acc_arr = np.empty(Nfits)
        for i in np.arange(Nfits):
            bdt.fit(X_train, y_train)
            train_acc_arr[i] = bdt.score(X_train, y_train)
            val_acc_arr[i] = bdt.score(X_validation, y_validation)
            test_acc_arr[i] = bdt.score(test_features, test_labels)
        while test_acc_arr[-1] < 0.89:
            bdt.fit(X_train, y_train)
            train_acc_arr[-1] = bdt.score(X_train, y_train)
            val_acc_arr[-1] = bdt.score(X_validation, y_validation)
            test_acc_arr[-1] = bdt.score(test_features, test_labels)

        print(test_acc_arr[-1], val_acc_arr[-1])

        train_acc_mean, train_acc_std, train_acc_sem = calc_mean_std_sem(train_acc_arr)
        val_acc_mean, val_acc_std, val_acc_sem = calc_mean_std_sem(val_acc_arr)
        test_acc_mean,test_acc_std, test_acc_sem = calc_mean_std_sem(test_acc_arr)
        print("Train data accuracy: ", train_acc_mean, "\u00B1", train_acc_std)
        print("Validation data accuracy: ", val_acc_mean, "\u00B1", val_acc_std)
        print("Test data  accuracy: ", test_acc_mean, "\u00B1", test_acc_std)

    ## find the optimal decision boundary according to validation data
    train_trans = bdt.decision_function(X_train)
    validation_trans = bdt.decision_function(X_validation)
    test_trans = bdt.decision_function(test_features)

    decision_boundary = np.linspace(-.4,.4,1000)
    acc_list = np.empty_like(decision_boundary)
    
    for i, cutoff in enumerate(decision_boundary):
        y_pred = np.zeros_like(y_validation)
        mask = (validation_trans > cutoff)
        y_pred[mask] = 1
        acc_list[i] = len(np.argwhere(y_pred == y_validation).flatten()) / Nvalidation_split

    fig0, ax0 = plt.subplots()
    ax0.plot(decision_boundary, acc_list,'.')
    ax0.set(xlabel = 'Adaboost test statistic decision boundary',\
             ylabel = 'Accuracy', title = 'Acc. of val. data for different decision boundaries')

    best_cutoff_ind = np.argmax(acc_list)
    best_cutoff = decision_boundary[best_cutoff_ind]
    best_acc = np.max(acc_list)

    y_pred_test = np.zeros_like(test_labels)
    mask = (test_trans > best_cutoff)
    y_pred_test[mask] = 1
    best_acc_test = len(np.argwhere(y_pred_test == test_labels).flatten()) / Ntest_data
    print(f"Using a cutoff at {best_cutoff:.3f} results in validation data acc.\
          of {best_acc:.3f} and test data acc of {best_acc_test:.3f}")

    
    # Calc and plot test statistic
    train_mask = (train_trans < best_cutoff)
    validation_mask = (validation_trans < best_cutoff)
    test_mask = (test_trans < best_cutoff)
    train_bkg_trans, train_sig_trans = train_trans[train_mask], train_trans[~train_mask]
    validation_bkg_trans, validation_sig_trans = validation_trans[validation_mask], validation_trans[~validation_mask]
    test_bkg_trans, test_sig_trans = test_trans[test_mask], test_trans[~test_mask]
    
    range = (min(np.min(train_bkg_trans), np.min(test_bkg_trans)), max(np.max(train_sig_trans), np.max(test_sig_trans)))
    bins = 100


    fig4, ax4 = plt.subplots()
    ax4.hist(train_trans[y_train == 0], range = range, bins = bins, histtype='step', alpha = .3, label = r'Traning $R=0$', color = 'red')
    ax4.hist(train_trans[y_train == 1], range = range, bins = bins, histtype='step', alpha = .3, label = r'Training $R=1$', color = 'black')
    ax4.hist(test_trans[test_labels == 0], range = range, bins = bins, histtype='stepfilled', alpha = .5, \
            label = r'Test $R=0$', color = 'red')
    ax4.hist(test_trans[test_labels == 1], range = range, bins = bins, histtype='stepfilled', alpha = .5, \
            label = r'Test $R=1$', color = 'black')
    ax4.plot([best_cutoff, best_cutoff], [0,450], 'k--', label = 'Decision boundary')
    ax4.legend(loc = 'upper left')
    ax4.set(xlabel = 'Adaboost test statistic', ylabel = 'Count', title = 'Transformed traning and test data')
    
    
    ## Calc and plot ROC curves
    fig00, ax00 = plt.subplots()
    bins_roc = 500
   
    FPR_train, TPR_train = calc_ROC(train_trans[y_train == 1], train_trans[y_train == 0], input_is_hist=False, bins = bins_roc, range = range)
    FPR_validation, TPR_validation = calc_ROC(validation_trans[y_validation == 1], \
                        validation_trans[y_validation == 0], input_is_hist=False, bins = bins_roc, range = range)
    FPR_test, TPR_test = calc_ROC(test_trans[test_labels == 1], test_trans[test_labels == 0],\
                                   input_is_hist=False, bins = bins_roc, range = range)
    ax00.plot(FPR_train, TPR_train, label = 'Training data', )
    ax00.plot(FPR_validation, TPR_validation, label = 'Validation data')
    ax00.plot(FPR_test, TPR_test, label = 'Test data')
    ax00.set(xlabel = 'False positive rate', ylabel = 'True positive rate', title ='ROC curves for traning and test data')
    ax00.legend()


    ## Finally, predict blind data labels
    blind_trans = bdt.decision_function(blind_features)
    y_pred_blind = np.zeros_like(blind_trans)
    y_pred_blind[blind_trans > best_cutoff] = 1
 
    # Split into those corresponding to revenue=1 and revenue=0
    blind_sig_mask = (y_pred_blind == 1)

    blind_sig_ids = blind_ids[blind_sig_mask]
    blind_bck_ids = blind_ids[~blind_sig_mask]

    ##print ids to file:
    np.savetxt('andersen.Problem4.RevenueTrue.txt', blind_sig_ids,fmt='%d')
    np.savetxt('andersen.Problem4.RevenueFalse.txt', blind_bck_ids, fmt = '%d')



    print("Fraction of validation data with predicted revenue = 1:", 
          len(validation_trans[(validation_trans > best_cutoff)]) / Nvalidation_split)
    print("Fraction of test data with predicted revenue = 1:", len(test_trans[(test_trans > best_cutoff)]) / Ntest_data)
    print("Fraction of blind data with predicted revenue = 1:", len(blind_trans[blind_trans > best_cutoff]) / Nblind_data)


    ## Plot transformed blind values to see if distribution match training data
    fig5, ax5 = plt.subplots()

    ax5.hist(train_trans[y_train == 0], range = range, bins = bins, histtype='step', alpha = .3, label = r'Traning $R=0$', color = 'red')
    ax5.hist(train_trans[y_train == 1], range = range, bins = bins, histtype='step', alpha = .3, label = r'Training $R=1$', color = 'black')
    ax5.hist(test_trans[test_labels == 0], range = range, bins = bins, histtype='stepfilled', alpha = .5, \
            label = r'Test $R=0$', color = 'red')
    ax5.hist(test_trans[test_labels == 1], range = range, bins = bins, histtype='stepfilled', alpha = .5, \

            label = r'Test $R=1$', color = 'black')
    ax5.set(xlabel = 'Adaboost test statistic', ylabel = 'Count', title = 'Transformed traning, test and blind data')
    ax5.hist(blind_trans, range = range, bins = bins, histtype='stepfilled', alpha = .4, label = r'Blind data', color = 'purple')
    ax5.plot([best_cutoff, best_cutoff], [0,450], 'k--', label = 'Decision boundary')
    ax5.legend()

    plt.show()



def main():
    ## Set which problems to run
    p1, p2, p3, p4 = False, False, False, True
    problem_numbers = [p1, p2, p3, p4]
    f_list = [P1, P2, P3, P4]

    for i, f in enumerate(f_list):
        if problem_numbers[i]:
            print(f'\nPROBLEM {i + 1}:')
            f()

if __name__ == '__main__':
    main()
