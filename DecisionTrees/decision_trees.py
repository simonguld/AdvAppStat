# Author: Simon Guldager Andersen
# Date (latest update): 

### SETUP ------------------------------------------------------------------------------------

## Imports:
import os, sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from iminuit import Minuit
from matplotlib import rcParams
from cycler import cycler

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from scipy import stats, integrate, interpolate, optimize
from tensorflow import keras
from keras import models, layers, optimizers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  #  {'0', '1', '2', '3'} = {Show all messages, remove info, remove info and warnings, remove all messages}
import tensorflow as tf
sys.path.append('Appstat2022\\External_Functions')
sys.path.append('AdvAppStat')
from ExternalFunctions import Chi2Regression, BinnedLH, UnbinnedLH
from ExternalFunctions import nice_string_output, add_text_to_ax    # Useful functions to print fit results on figure
from statistics_helper_functions import calc_ROC, calc_fisher_discrimminant, calc_ROC_AUC


## Change directory to current one
os.chdir('AdvAppStat\DecisionTrees')


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
    exc1, exc2, exam2016 = False, True, False

    if exc1: 
        ##EXC1: We are supplied with 3 variable background and signal training and test data.
        ## Ensure that traning and test data are similarly distributed
        ## Goal: Make the best possible classicification and quantify what that is.
        ## Ensure / measure loss function and avoid overtraining

        train_sig = np.loadtxt('BDT_signal_train.txt')
        train_bkg = np.loadtxt('BDT_background_train.txt')

        test_sig = np.loadtxt('BDT_signal_test.txt')
        test_bkg = np.loadtxt('BDT_background_test.txt')

        Nparameters = len(train_sig[0,:])

        ## Plot 1D hists of training vs test data for bck and sig
        fig1, ax1 = plt.subplots(ncols = 3, figsize = (18,6))
        ax1 = ax1.flatten()
        bins = 50

        for i, ax in enumerate(ax1):
            ax.hist(train_sig[:,i], bins = bins, histtype = 'stepfilled', color = 'red', alpha = .3, label = 'Train sig')
            ax.hist(train_bkg[:,i], bins = bins, histtype = 'stepfilled', color = 'black', alpha = .3, label = 'Train bkg')

            ax.hist(test_sig[:,i],bins = bins, histtype = 'step', color = 'red', alpha = .3, label = 'Test sig')
            ax.hist(test_bkg[:,i], bins = bins,histtype = 'step', color = 'black', alpha = .3, label = 'Test bkg')
            ax.set(xlabel = f'x{i}', ylabel = 'count')
            ax.legend()

        ## Make scatter plots of traning data
        fig2, ax2 = plt.subplots(ncols = 3, figsize = (18,6))
        ax2 = ax2.flatten()

        for i in np.arange(Nparameters):
            ax2[i].scatter(train_bkg[:,i], train_bkg[:,(i+1) % Nparameters], label = 'Bck', color = 'black', s = 5, alpha = .5)
            ax2[i].scatter(train_sig[:,i], train_sig[:,(i+1) % Nparameters], label = 'sig', color = 'red', s = 5, alpha = .5)
            ax2[i].set(xlabel = f'x{i}', ylabel = f'x{(i+1) % Nparameters}')
            ax2[i].legend()


        fig1.tight_layout()
        fig2.tight_layout()


        ## Train boosted decision tree (BDT) to classify training data.
        train_data = np.r_['0,2', train_sig, train_bkg]
        train_labels = np.r_['0', np.ones_like(train_sig[:,0]), np.zeros_like(train_bkg[:,0])]

        test_data =  np.r_['0,2', test_sig, test_bkg]
        test_labels = np.r_['0', np.ones_like(test_sig[:,0]), np.zeros_like(test_bkg[:,0])]

        
        ## Decide which scheme to run
        GB, ADA = False, True
        if GB:
            ## Set Graident boosting parameters
            kwargs_BDT = {'loss': "log_loss", 'n_estimators': 50, 'learning_rate': 1, 'max_depth': 4, 'criterion': "friedman_mse", \
                          'min_samples_leaf': 1, 'min_samples_split': 2, 'subsample': 1.0, 'max_features': None,\
                              'n_iter_no_change': None, 'validation_fraction': 0.1, 'tol': 0.0001, 'ccp_alpha':  0.0}
            bdt = GradientBoostingClassifier(**kwargs_BDT)
        if ADA:
            ## Set AdaBoost parameters
            kwargs_BDT = {'n_estimators': 46, 'learning_rate': 1.3} # default are 50,1
            ## Set decision tree parameters
            kwargs_DT = { 'max_depth': 8, 'criterion': "gini", 'min_samples_split': 2, 'min_samples_leaf': 1, \
                        'splitter': "best", 'max_features': None, 'ccp_alpha':  0.0} 
            # default are 1, gini, 2, 1, best, 0. For all parameters, use BDT.get_params()    
            bdt = AdaBoostClassifier(DecisionTreeClassifier(**kwargs_DT), **kwargs_BDT)

        ## Find optimal value of the following parameters:
       
      #  n_estimator_range = np.arange(5,50)
        param_grid = {'n_estimators': list(np.arange(46,49,1).astype('int')),'base_estimator__max_depth': list(np.arange(8,9,1).astype('int')),\
                     'learning_rate': [1.3], 'base_estimator__ccp_alpha': list(np.linspace(0,0.2,10))}
        #del kwargs_BDT['n_estimators']

        bdt = AdaBoostClassifier(DecisionTreeClassifier(**kwargs_DT), **kwargs_BDT)

        # use grid
     #   grid_cv = GridSearchCV(bdt, param_grid, scoring = 'accuracy', cv = 5, n_jobs = -1).fit(train_data, train_labels)
        # use randomized
        grid_cv = RandomizedSearchCV(bdt, param_grid, n_iter = 10, scoring = 'accuracy', cv = 5, n_jobs = -1).fit(train_data, train_labels)

        print("Param for GS", grid_cv.best_params_)
        print("CV score for GS", grid_cv.best_score_)
        #print(grid_cv.cv_results_)
        bdt.fit(train_data, train_labels)
        #y_predict = bdt.predict(test_data)
        # Calc average accurarcy
        acc = bdt.score(test_data, test_labels)
        print("Accuracy of test data predictions: ", acc)

        ## Calc. the test statistic
        bkg_trans_train, sig_trans_train = bdt.decision_function(train_bkg), bdt.decision_function(train_sig)
        bkg_trans_test, sig_trans_test = bdt.decision_function(test_bkg), bdt.decision_function(test_sig)

        ## Calc and plot ROC curves 
        fig0, ax0 = plt.subplots()
        bins_roc = 500
        range = (min(np.min(bkg_trans_train), np.min(bkg_trans_test)), max(np.max(sig_trans_train), np.max(sig_trans_test)))

        FPR_train, TPR_train = calc_ROC(sig_trans_train, bkg_trans_train, input_is_hist=False, bins = bins_roc, range = range)
        FPR_test, TPR_test = calc_ROC(sig_trans_test, bkg_trans_test, input_is_hist=False, bins = bins_roc, range = range)
        ## calc areas under curve
        AUC_train, AUC_test = calc_ROC_AUC(FPR_train, TPR_train), calc_ROC_AUC(FPR_test, TPR_test)
        print("AUC for train, test: ", AUC_train, AUC_test )
        ax0.plot(FPR_train, TPR_train, label = 'Training data', )
        ax0.plot(FPR_test, TPR_test, label = 'Test data')
        ax0.set(xlabel = 'False positive rate', ylabel = 'True positive rate', title ='ROC curves for test and training data')
        ax0.legend()


        
        

        fig3, ax3 = plt.subplots()

        ax3.hist(bkg_trans_train, bins = bins, histtype='step', alpha = .3, label = 'transformed bck train', color = 'red')
        ax3.hist(sig_trans_train, bins = bins, histtype='step', alpha = .3, label = 'transformed sig train', color = 'black')
        ax3.hist(bkg_trans_test, bins = bins, histtype='stepfilled', alpha = .3, label = 'transformed bck test', color = 'red')
        ax3.hist(sig_trans_test, bins = bins, histtype='stepfilled', alpha = .3, label = 'transformed sig test', color = 'black')
        ax3.legend()

    if exc2:
        ## EXC2: Download data set of 16 variables. 1st column is labels. 
        # All even rows are signal, odd are background

        ## Split into test training.
        ## optimize accuracy etc. possibly using ROC curves.
        ## Try getting rid of the least important features. What happens?

        data = np.loadtxt('BDT_16var.txt')[:,1:]
        Npoints, Nfeatures = data.shape
        bins = 50
 
        data_sig = data[np.arange(0, Npoints, 2),:]
        data_bkg = data[np.arange(1, Npoints, 2),:]

    
        ## 1 is signal
        labels = np.zeros_like(data[:,0])
        labels[np.arange(0, Npoints, 2)] = 1

        ## Split into test and training data
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.20)

        train_signal_mask = (y_train == 1)
        test_signal_mask = (y_test == 1)

        print("Nsignal and Nbck: ", data_sig.shape[0], data_bkg.shape[0])

        if 0:
            fig4, ax4 = plt.subplots(ncols = 4, nrows = 4, figsize = (24,24))
            ax4 = ax4.flatten()

            for i, ax in enumerate(ax4):
                if i == 11:
                    range = (0,200)
                    ax.hist(data_bkg[:,i], range = range, bins = bins, histtype = 'stepfilled', color = 'red', alpha = .3, label = 'Train sig')
                    ax.hist(data_sig[:,i], range = range, bins = bins, histtype = 'stepfilled', color = 'black', alpha = .3, label = 'Train bkg')
                else:
                    ax.hist(data_bkg[:,i], bins = bins, histtype = 'stepfilled', color = 'red', alpha = .3, label = 'Train sig')
                    ax.hist(data_sig[:,i], bins = bins, histtype = 'stepfilled', color = 'black', alpha = .3, label = 'Train bkg')
                
                ax.set(xlabel = f'x{i}', ylabel = 'count')
                #ax.legend()


         ## Decide which scheme to run
        GB, ADA = False, True

        if GB:
            ## Set Graident boosting parameters
            kwargs_BDT = {'loss': "log_loss", 'n_estimators': 50, 'learning_rate': 1, 'max_depth': 4, 'criterion': "friedman_mse", \
                          'min_samples_leaf': 1, 'min_samples_split': 2, 'subsample': 1.0, 'max_features': None,\
                              'n_iter_no_change': None, 'validation_fraction': 0.1, 'tol': 0.0001, 'ccp_alpha':  0.0}
            bdt = GradientBoostingClassifier(**kwargs_BDT)
        if ADA:
            ## Set AdaBoost parameters
            kwargs_BDT = {'n_estimators': 50, 'learning_rate': 1} # default are 50,1
            ## Set decision tree parameters
            kwargs_DT = { 'max_depth': 2, 'criterion': "gini", 'min_samples_split': 2, 'min_samples_leaf': 1, \
                        'splitter': "best", 'max_features': None, 'ccp_alpha':  0.0} 
            # default are 1, gini, 2, 1, best, 0. For all parameters, use BDT.get_params()    
            bdt = AdaBoostClassifier(DecisionTreeClassifier(**kwargs_DT), **kwargs_BDT)


         #  n_estimator_range = np.arange(5,50)
        param_grid = {'n_estimators': list(np.arange(45,46).astype('int')),'base_estimator__max_depth': [2],\
                     'learning_rate': list(np.arange(0.45,0.46,0.05)), 'base_estimator__max_features': [12]} #, 'base_estimator__ccp_alpha': list(np.linspace(0,0.2,10))}
        #del kwargs_BDT['n_estimators']

        bdt = AdaBoostClassifier(DecisionTreeClassifier(**kwargs_DT), **kwargs_BDT)
      
        # use grid
     #   grid_cv = GridSearchCV(bdt, param_grid, scoring = 'accuracy', cv = 5, n_jobs = -1).fit(train_data, train_labels)
        # use randomized
        bdt = RandomizedSearchCV(bdt, param_grid, n_iter = 30, scoring = 'accuracy', cv = 5, n_jobs = -1).fit(X_train, y_train)
        
        ## Fit using best parameters
        #bdt = AdaBoostClassifier(DecisionTreeClassifier, **grid_cv.best_params_)
        #bdt.fit(X_train, y_train)

        print("Best Param for GS", bdt.best_params_)
        print("CV score for GS", bdt.best_score_)
        #print(grid_cv.cv_results_)
    

        acc = bdt.score(X_test, y_test)
        print("validation data  acc: ", acc)

       # print("Feature importance: ", bdt.feature_importances_)
       # print(bdt.train_score_)
       # print(bdt.n_features_in_)
        # Calc and plot test statistic
        train_bkg_trans = bdt.decision_function(X_train[~train_signal_mask])
        train_sig_trans = bdt.decision_function(X_train[train_signal_mask])
        test_bkg_trans = bdt.decision_function(X_test[~test_signal_mask])
        test_sig_trans = bdt.decision_function(X_test[test_signal_mask])
        range = (-50,50)
        range = (min(np.min(train_bkg_trans), np.min(test_bkg_trans)), max(np.max(train_sig_trans), np.max(test_sig_trans)))

        bins = 100


        fig4, ax4 = plt.subplots()
        ax4.hist(train_bkg_trans, range = range, bins = bins, histtype='step', alpha = .3, label = 'transformed bck', color = 'red')
        ax4.hist(train_sig_trans, range = range, bins = bins, histtype='step', alpha = .3, label = 'transformed sig', color = 'black')
        ax4.hist(test_bkg_trans, range = range, bins = bins, histtype='stepfilled', alpha = .3, label = 'test bck', color = 'red')
        ax4.hist(test_sig_trans, range = range, bins = bins, histtype='stepfilled', alpha = .3, label = 'test sig', color = 'black')
        ax4.legend()

        ## Calc and plot ROC curves
        fig00, ax00 = plt.subplots()
        bins_roc = 500
        #range = (min(np.min(train_bkg_trans), np.min(test_bkg_trans)), max(np.max(train_sig_trans), np.max(test_sig_trans)))

        FPR_train, TPR_train = calc_ROC(train_sig_trans, train_bkg_trans, input_is_hist=False, bins = bins_roc, range = range)
        FPR_test, TPR_test = calc_ROC(test_sig_trans, test_bkg_trans, input_is_hist=False, bins = bins_roc, range = range)

        ax00.plot(FPR_train, TPR_train, label = 'Training data', )
        ax00.plot(FPR_test, TPR_test, label = 'Test data')
        ax00.set(xlabel = 'False positive rate', ylabel = 'True positive rate', title ='ROC curves for test and training data')
        ax00.legend()

    if exam2016:
        ## We consider data having 9 feautres, with last row a label (2 = benign, 4 = malignant)
        # Goal: Train architechture and predict labels for test data
        ## possibly try several classifiers
        
        train_data = np.loadtxt('breast-cancer-wisconsin_train-test.txt')[:,1:]
        train_labels = train_data[:,-1].astype('float')
        train_data = train_data[:,:-1]
        test_data = np.r_['0,2', np.loadtxt('benign_true.txt')[:,1:], np.loadtxt('malignant_true.txt')[:,1:]]
        test_labels = test_data[:,-1].astype('float')
        test_data = test_data[:,:-1]


        ## Split training data for cross validation purposes
        X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.15)

        train_signal_mask = (y_train == 4)
        test_signal_mask = (y_test == 4)

        ###---------------------------------------------------------
        ## Classify using Fisher discriminant
        fisher_sig, fisher_bck, weights = calc_fisher_discrimminant(train_data[train_labels == 4], train_data[train_labels == 2], weight_normalization=100)

        ## Plot
        fig0, ax0 = plt.subplots()
        bins = 50
        ax0.hist(fisher_bck, bins = bins, histtype='step', alpha=.3, label = 'Train benign')
        ax0.hist(fisher_sig, bins = bins, histtype='step', alpha=.3, label = 'Train malig')
      

        ## Define cutoff 
        x_cutoff = 6.0
       
        ## Use weights to calc fisher discrimminants for test data
        test_fisher =  np.sum((weights) * test_data, axis = 1) 
        ax0.hist(test_fisher, bins = bins, histtype='stepfilled', alpha=.3, label = 'Test')
        
        
        ## find benign and malignant indicies
        fisher_signal_mask = (test_fisher >= x_cutoff)
        fisher_pred_labels = 2 * np.ones_like(test_labels)
        fisher_pred_labels[fisher_signal_mask] = 4

        fisher_accuracy = len(np.argwhere(fisher_pred_labels - test_labels == 0).flatten()) / len(test_labels)
        print("Fisher accuracy: ", fisher_accuracy)

        # CALC ROC curve
        bins = 50
        range = (np.min(test_fisher), np.max(test_fisher))
        fisher_pred_sig = np.sum((weights) * test_data[(test_labels == 4)], axis = 1)
        fisher_pred_bck = np.sum((weights) * test_data[(test_labels == 2)], axis = 1)
        fisherFPR, fisherTPR = calc_ROC(fisher_pred_sig, fisher_pred_bck, \
                                                  input_is_hist=False, bins = 10 * bins, range = range)

        ax0.hist(fisher_pred_sig, bins = bins, histtype='stepfilled', alpha=.3, label = 'Test pred sig')
        ax0.hist(fisher_pred_bck, bins = bins, histtype='stepfilled', alpha=.3, label = 'Test pred bck')
        ax0.legend()

        ##-------------------------------------------------------------------------------------------------------------------------------- 

        ##------------------------------
        # DO KNN:

        # Set parameters (algorithm can be Balltree og KDTree or brute force, p is power of L_p norm, leaf_size,....
        # weights can be uniform (all nearest neighbors count the same) or distance (NN-weight scales with inverse distance to evaluation point)
        kwargs_knn = {'algorithm': 'brute',  'n_neighbors': 7, 'leaf_size': 30, 'metric': 'minkowski', 'p': 1, 'weights': 'uniform'}
    
        knn = KNeighborsClassifier(**kwargs_knn)

        ## Build knn tree
        knn.fit(X_train, y_train)

        ## Predicts cross reference data labels
        knn_cr_labels = knn.predict(X_test)
        mask = (knn_cr_labels - y_test == 0)
        knn_cr_accuracy = len(knn_cr_labels[mask]) / len(y_test)
        ## Predict test data labels
        knn_pred_labels = knn.predict(test_data)
        print(knn_pred_labels[:30])
        mask = (knn_pred_labels - test_labels == 0)
        knn_accuracy = len(knn_pred_labels[mask]) / len(test_labels)

    
        print("knn cross ref accuracy: ", knn_cr_accuracy)
        print("knn test accuracy: ", knn_accuracy)


        ##-----------------------------
        # Do neural network

        ## normalize all variables


         # Build CNN model
        model = models.Sequential()

        #1st convolution
        #model.add(layers.Conv2D())

        layers.Normalization(axis = 0)

        model.add(layers.Dense(9, activation = 'relu', input_shape = (len(train_data[0,:]),) ))
      #  model.add(layers.Dense(9, activation = 'relu'))
       # model.add(layers.Dense(9, activation = 'relu'))
        model.add(layers.Dense(1, activation = 'sigmoid')) #, activation = 'sigmoid'))
        
        model.build()
        print(model.summary())
      

        # Compile
        """
        Model.compile(
        optimizer="rmsprop",
        loss=None,
        metrics=None,
        loss_weights=None,
        weighted_metrics=None,
        run_eagerly=None,
        steps_per_execution=None,
        jit_compile=None,
        **kwargs
    )
        """
        kwargs_ann = {'optimizer': 'adam', 'loss': 'binary_crossentropy', 'metrics': ['acc'], \
                      }   # 'loss': tf.keras.losses.BinaryCrossentropy(from_logits=True),
        model.compile(**kwargs_ann)

        # Train model
        y_train_ann = 0.5 * y_train.astype('int') - 1
        y_test_ann = 0.5 * y_test.astype('int') - 1
        
        history = model.fit(X_train, y_train_ann, epochs=150, 
                        validation_data=(X_test, y_test_ann), verbose = 0)

        fig5,ax5 = plt.subplots()
        loss, val_loss = history.history['loss'], history.history['val_loss']
        iterations = np.arange(len(loss))
        ax5.plot(iterations, loss, '^',label = 'Training loss')
        ax5.plot(iterations, val_loss,'s', label = 'Validation loss')
        ax5.legend()
      
        y_pred = model.predict(test_data, batch_size = 32, verbose = 1)
        y_pred_bool = ((1 + np.round(y_pred)) * 2).flatten()

        
        ## calc ann accuracy
        ann_mask = (y_pred_bool - test_labels == 0)

        ann_acc = len(y_pred_bool[ann_mask]) / len(test_labels)
                                                   
        print("cnn accuracy: ", ann_acc)



        ##-------------------------------------
 
        ## Build and apply boosted tree
        GB, ADA = True, False

        if GB:
            ## Set Graident boosting parameters
            kwargs_BDT = {'loss': "log_loss", 'n_estimators': 50, 'learning_rate': 1, 'max_depth': 4, 'criterion': "friedman_mse", \
                          'min_samples_leaf': 1, 'min_samples_split': 2, 'subsample': 1.0, 'max_features': None,\
                              'n_iter_no_change': None, 'validation_fraction': 0.1, 'tol': 0.0001, 'ccp_alpha':  0.0}
            bdt = GradientBoostingClassifier(**kwargs_BDT)
        if ADA:
            ## Set AdaBoost parameters
            kwargs_BDT = {'n_estimators': 50, 'learning_rate': 1} # default are 50,1
            ## Set decision tree parameters
            kwargs_DT = { 'max_depth': 2, 'criterion': "gini", 'min_samples_split': 2, 'min_samples_leaf': 1, \
                        'splitter': "best", 'max_features': None, 'ccp_alpha':  0.0} 
            # default are 1, gini, 2, 1, best, 0. For all parameters, use BDT.get_params()    
            bdt = AdaBoostClassifier(DecisionTreeClassifier(**kwargs_DT), **kwargs_BDT)


        bdt.fit(X_train, y_train)
        acc = bdt.score(X_test, y_test)
        print("BDT Cross validation accuracy: ", acc)

        print("Feature importance: ", bdt.feature_importances_)
       # print(bdt.train_score_)
       # print(bdt.n_features_in_)
        # Calc and plot test statistic
        train_bkg_trans = bdt.decision_function(X_train[~train_signal_mask])
        train_sig_trans = bdt.decision_function(X_train[train_signal_mask])
        test_bkg_trans = bdt.decision_function(X_test[~test_signal_mask])
        test_sig_trans = bdt.decision_function(X_test[test_signal_mask])
        range = (-50,50)
        range = (min(np.min(train_bkg_trans), np.min(test_bkg_trans)), max(np.max(train_sig_trans), np.max(test_sig_trans)))

        bins = 100

        fig4, ax4 = plt.subplots()
        ax4.hist(train_bkg_trans, range = range, bins = bins, histtype='step', alpha = .3, label = 'train bck', color = 'red')
        ax4.hist(train_sig_trans, range = range, bins = bins, histtype='step', alpha = .3, label = 'train sig', color = 'black')
        ax4.hist(test_bkg_trans, range = range, bins = bins, histtype='stepfilled', alpha = .3, label = 'CV bck', color = 'red')
        ax4.hist(test_sig_trans, range = range, bins = bins, histtype='stepfilled', alpha = .3, label = 'CV sig', color = 'black')
        

        ## Calc and plot ROC curves
        fig00, ax00 = plt.subplots()
        bins_roc = 500
        #range = (min(np.min(train_bkg_trans), np.min(test_bkg_trans)), max(np.max(train_sig_trans), np.max(test_sig_trans)))

        FPR_train, TPR_train = calc_ROC(train_sig_trans, train_bkg_trans, input_is_hist=False, bins = bins_roc, range = range)
        FPR_test, TPR_test = calc_ROC(test_sig_trans, test_bkg_trans, input_is_hist=False, bins = bins_roc, range = range)

        ax00.plot(FPR_train, TPR_train, label = 'Training data', )
        ax00.plot(FPR_test, TPR_test, label = 'Cross-validation data')
        ax00.set(xlabel = 'False positive rate', ylabel = 'True positive rate', title ='ROC curves for test and training data')
       


        ## Predict test data labels
       # Calculate accuracy of test data predictions

        test_accuracy = bdt.score(test_data, test_labels)
        test_mask = (test_labels == 4)
        print("BDT Test data accuracy: ", test_accuracy)

        ## Calc and plot test statistics and ROC
        true_bkg_trans = bdt.decision_function(test_data[~test_mask])
        true_sig_trans = bdt.decision_function(test_data[test_mask])
        range = (min(np.min(train_bkg_trans), np.min(test_bkg_trans)), max(np.max(train_sig_trans), np.max(test_sig_trans)))

        ax4.hist(true_bkg_trans, range = range, bins = bins, histtype='stepfilled', alpha = .3, label = 'transformed bck', color = 'plum')
        ax4.hist(true_sig_trans, range = range, bins = bins, histtype='stepfilled', alpha = .3, label = 'transformed sig', color = 'grey')

        FPR_pred, TPR_pred = calc_ROC(true_sig_trans, true_bkg_trans, input_is_hist=False, bins = bins_roc, range = range)
       
        ax00.plot(FPR_pred, TPR_pred, label = 'BDT', )
        ax00.plot(fisherFPR, fisherTPR, label = 'Fisher')

        ax00.legend()
        ax4.legend()
    plt.show()




if __name__ == '__main__':
    main()
