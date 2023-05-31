# Author: Simon Guldager Andersen
# Date (latest update): 24-03-2023

### SETUP ------------------------------------------------------------------------------------

## Imports:
import os, sys
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from iminuit import Minuit
from matplotlib import rcParams
from cycler import cycler
from scipy import stats, integrate, interpolate, optimize

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  #  {'0', '1', '2', '3'} = {Show all messages, remove info, remove info and warnings, remove all messages}
import tensorflow as tf
from tensorflow import keras
from keras import models, layers, optimizers, regularizers
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_log_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn import metrics
from category_encoders import LeaveOneOutEncoder, TargetEncoder
import eli5    
from eli5.sklearn import PermutationImportance


sys.path.append('Appstat2022\\External_Functions')
sys.path.append('AdvAppStat')
from ExternalFunctions import Chi2Regression, BinnedLH, UnbinnedLH
from ExternalFunctions import nice_string_output, add_text_to_ax    # Useful functions to print fit results on figure
from statistics_helper_functions import calc_ROC, calc_fisher_discrimminant, calc_ROC_AUC


## Change directory to current one
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

## Set plotting style and print options
sns.set_theme()
sns.set_style("darkgrid")
sns.set_context("paper") #Possible are paper, notebook, talk and poster
# Set figure format
wide_format, square_format = False, True
if wide_format:
    d = {'lines.linewidth': 2, 'axes.titlesize': 18, 'axes.labelsize': 18, 'xtick.labelsize': 12, 'ytick.labelsize': 12,\
     'legend.fontsize': 15, 'font.family': 'serif', 'figure.figsize': (9,6)}
if square_format:
    d = {'lines.linewidth': 2, 'axes.titlesize': 16, 'axes.labelsize': 14, 'xtick.labelsize': 10, 'ytick.labelsize': 10,\
     'legend.fontsize': 12, 'font.family': 'serif', 'figure.figsize': (6,6)}
    
d_colors = {'axes.prop_cycle': cycler(color = ['teal', 'navy', 'coral', 'plum', 'purple', 'olivedrab',\
         'black', 'red', 'cyan', 'yellow', 'khaki','lightblue'])}
rcParams.update(d)
rcParams.update(d_colors)
np.set_printoptions(precision = 5, suppress=1e-10)


### FUNCTIONS ----------------------------------------------------------------------------------

def extract_sublist(lst, index):
    return list(list(zip(*lst))[index])

def outliers_detection(df,f, cutoff = 1.5):
        Q1 = np.percentile(df[f],25)
        Q3 = np.percentile(df[f],75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - (cutoff * IQR)
        upper_bound = Q3 + (cutoff * IQR)
        
        ls = df.index[(df[f] < lower_bound ) | (df[f] > upper_bound) ]
        
        return ls
    
def remove_outliers(data_frame, features, cutoff = 1.5):
    index = []

    for feature in features:
        index.extend(outliers_detection(data_frame, feature, cutoff = cutoff))

    return data_frame.drop(index)

def data_cleaning(data, verbose = True):
    ### Clean data

    Npoints, _ = data.shape
    if verbose:
        print("No. of data points for raw data: ", Npoints)

    ### STEP1: Remove duplicated points
    duplicated_rows_indices = data.index[data.duplicated()]
    data = data.drop(duplicated_rows_indices)

    Nnew, _ = data.shape
    if verbose:
        print("No. of data points after removing duplicates: ", Nnew)

    ### STEP2: Replace the missing Levy entries with Knn estimate
    # Replace missing values with nan
    data['Levy'].replace({'-': np.nan}, inplace = True)
    data['Levy'] = data['Levy'].astype('float')
    # intialize imputer and impuate missing values
    imputer = KNNImputer(n_neighbors = 5, weights = 'uniform', metric = 'nan_euclidean')
    data['Levy'] = imputer.fit_transform(data['Levy'].values.reshape(-1,1))
    
    ## STEP3: Door values are 02-Mar, 04-May, >5. Convert to (2,4,6))
    data['Doors'] = data['Doors'].replace('02-Mar','2')
    data['Doors'] = data['Doors'].replace('04-May','4')
    data['Doors'] = data['Doors'].replace('>5', '6')

    ## STEP4: 
    # remove unit from mileage column
    data['Mileage'] =  data['Mileage'].str.extract('(\d+)').astype(float)
   
    ## STEP5: create separate column for turbo and convert engine volume to float
    # make sure engine size is with one decimal
    data['Turbo'] = data['Engine volume'].str.contains("Turbo")
    data['Turbo'] = data['Turbo'].map({False:0, True: 1})
    data['Engine volume'] = data['Engine volume'].str.split().str[0]

    ## STEP 5.5: Convert binary categories to floats
    data['Wheel'] = data['Wheel'].map({'Left wheel': 0, 'Right-hand drive': 1})
    data['Leather interior'] = data['Leather interior'].map({'No': 0, 'Yes': 1})
   
    ## STEP 6: convert integers to floats
    num_columns = ['Price', 'Airbags', 'Prod. year', 'Engine volume', 'Doors', 'Turbo', 'Wheel', 'Leather interior']
    for column in num_columns:
        data[column] = data[column].astype('float')

    ## STEP 7: Group infrequent labels together
    minimum_count = 3
    columns_to_bundle = ['Manufacturer', 'Model']
    for i, column in enumerate(columns_to_bundle):
        unique_count = data.groupby(column).count()['ID'].sort_values(ascending=False)
        for value in unique_count.index:
            if unique_count[value] < minimum_count:
                data[column] = data[column].replace(value,'OTHER')

    # STEP 8: Abbreviate long names
    data.rename(columns ={'Manufacturer': 'Manufac.', 'Leather interior': 'Leather', 'Engine volume': 'Eng. vol.', \
                          'Gear box type': 'Gear box', 'Drive wheels': 'Dr. Wheel'}, inplace = True)

    ### STEP 9: Remove ID column
    data = data.drop(columns = 'ID')

    if verbose:
        ## examine values of each column
        for column in data.columns:       
        # print(column)
            print(f' No. of unique values in column {column}:', len(data[column].unique()))
            #print(data[column].unique())

        print("No. of unique points after cleaning: ", data.shape[0])
    return data

def visualize_data(data, prefix = 'training', price_hist = False, numerical_plots = False, categorical_plots = False):
    ## Step 1: Plot price as a function of numeric variables

    if numerical_plots:
            for column in data.select_dtypes(include='float').columns:
                    if column == 'Price':
                        continue
                    else:
                        plt.figure()
                        d={}

                        plt.scatter(data[column], data['Price'], s = 3)
                
                        plt.title(f'Price as a function of {column} for {prefix} data')
                        plt.ylabel('Price')
                        plt.xlabel(f'{column}')
                        plt.show()
    if categorical_plots:
        ## Step 2: Plot price against categorical data
        for column in data.select_dtypes(include='object').columns:
                plt.figure()
                d={}

                for label in data[column].unique():
                    indices = data.index[data[column] == label]
                    prices = data['Price'][indices]
                    #d.update({f'{label}': [prices.mean(), prices.std(ddof = 1)]})
                    d.update({f'{label}': prices})

                for label, values in zip(d.keys(), d.values()):
                    plt.scatter([label] * len(values), values, color = 'purple', s = 3)
                if len(d.keys()) > 15:
                    fontsize = 6
                    rotation = -45
                elif 4 < len(d.keys()) <= 10:
                    fontsize = 9
                    rotation = -45
                else:
                    fontsize = 12
                    rotation = 0
                plt.title(f'Price as a function of {column} for {prefix} data')
                plt.ylabel('Price')
                plt.xticks(fontsize = fontsize, rotation = rotation)
                plt.show()
    if price_hist:
        fig0, ax0 = plt.subplots()
        range = (0, 80_000)
        bins = 160
        ax0.hist(data['Price'], bins = bins, range = range, histtype='stepfilled', alpha=.3)
        ax0.set(xlabel = 'Price distribution', ylabel = 'Count', title = f'Histogram of {prefix} data prices')
        plt.show()

def do_label_encoding(dataframe):
    """
    Transforms each value for each categorical features into integer. If there are 10 categories in columns, they will be transformed to numbers 0-9
    """
    data = dataframe.copy()
    target_columns = data.select_dtypes(include='object').columns
    labelencoder = LabelEncoder()

    for col in target_columns:
        data[f'{col}'] = labelencoder.fit_transform(data[col])
    
  #  data = data.drop(columns = target_columns)
    return data

def do_target_encoding(input_arr, leave_one_out = True, smoothing = 10):
    data = input_arr.copy()
    
    if leave_one_out:
        encoder = LeaveOneOutEncoder()
    else:
        encoder = TargetEncoder(smoothing = smoothing)

    target_categories = data.select_dtypes(include='object').columns

    for column in target_categories:
        data[column] = encoder.fit_transform(data[column], data['Price'])

    return data

def do_mix_encoding(dataframe, drop_first = 'True',  transition_from_dummy_to_target_encoding_cutoff = 10,\
                     leave_one_out = True, smoothing = 10): 
    """
    Transform all categorical variables with less than transition_from_dummy_to... cutoff by dummy encdoing,
    and all above the cutoff by target encoding
    """
 
    data = dataframe.copy()
    # Extract categorical columns
    categorical_columns = data.select_dtypes(include='object').columns

    target_columns_dummy = []
    target_columns_target = []
    for column in categorical_columns:
        if len(data[column].unique()) >= transition_from_dummy_to_target_encoding_cutoff:
            target_columns_target.append(column)
        else:
            target_columns_dummy.append(column)

    # Do target encoding
    if leave_one_out:
        encoder = LeaveOneOutEncoder()
    else:
        encoder = TargetEncoder(smoothing = smoothing)

    for column in target_columns_target:
        data[column] = encoder.fit_transform(data[column], data['Price'])

    # Build dummy index dataframe
    dummy_columns = pd.get_dummies(data[target_columns_dummy], drop_first=drop_first).astype('float')

    ## Append new columns to dataframe
    data = pd.concat([data, dummy_columns], axis = 1)

    ## Drop original columns
    data = data.drop(columns = target_columns_dummy)
    return data

def do_dummy_encoding(dataframe, drop_first = 'False'):
    """
    For each cateogrical column with categories (l1,...,ln), this function transform the category into n ( if drop_first = False)
    or n-1 columns, where [0,0,...,0] represents l1,
    [1,0,...,0] rep. l2, [0,1,0,...,0] rep l3 etc
    """
    data = dataframe.copy()
    # Extract categorical columns
    target_columns = data.select_dtypes(include='object').columns
    # Build dummy index dataframe
    dummy_columns = pd.get_dummies(data[target_columns], drop_first=drop_first)

    ## Append new columns to dataframe
    data = pd.concat([data, dummy_columns], axis = 1)

    ## Drop original columns
    data = data.drop(columns = target_columns)
    return data

def rescale_input_linear(dataframe, target_interval = [- 1, 1]):
    data = dataframe.copy()
    # define the rescaling function
    def rescaler_linear(x, target_interval = target_interval):
        xmin, xmax = np.min(x), np.max(x)
        ymin, ymax = target_interval[0], target_interval[1]

        a = (ymax - ymin) / (xmax - xmin)
        b = ymin - a * xmin
        return b + a * x
    
    # Extract columns to rescale
    target_columns = data.select_dtypes(include = ['float64'])

    for col in target_columns:
        if col == 'Price':
            continue
        else:
            data[col] = rescaler_linear(data[col])
    return data

def plot_feature_importance(perm, feature_names, fig, ax):
 
        vals = perm.feature_importances_
        stds = perm.feature_importances_std_
        names = feature_names
        sorting_ind = np.argsort(vals)
        valmax = max(vals)
        vals = np.flip(vals[sorting_ind]) / max(vals)
        names = np.flip(names[sorting_ind])
        stds = np.flip(stds[sorting_ind]) / valmax
    
        fig, ax = plt.subplots()
        ax.barh(list(names), vals, alpha = .3)
        ax.invert_yaxis()
        plt.yticks(fontsize = 9)
        ax.set(xlabel = 'Average rel. feature importance', title = 'Relative feature importance for neural network')
        ax.errorbar(vals, list(names), xerr = stds, fmt='k.', capsize = 1, capthick = 1, elinewidth = 1)
 
def evaluate_feature_dropping(base_model, X_train, X_validation, y_train, y_validation,\
                               Niterations = 1, Nepochs = 1, batch_size = 128, feature_names = None):
        # Extract feature names 
        if feature_names is None:
            feature_names = X_train.columns
    
        Nfeatures = len(X_train.columns)
        Ndrop = len(feature_names)
        ## initialize loss arrays. The last column is for baseline losses with all feautures present
        train_loss = np.empty([Niterations, Ndrop + 1], dtype = 'float')
        val_loss = np.empty_like(train_loss)

        for n in np.arange(Niterations):
            # evaluate baseline performance
            model = base_model(input_shape = (Nfeatures,))
            history = model.fit(X_train, y_train, epochs= Nepochs, batch_size = batch_size, validation_data=(X_validation, y_validation), verbose = 0)
            # store results
            train_loss[n, -1] = history.history['loss'][-1]
            val_loss[n, -1] = history.history['val_loss'][-1]
            for i, column in enumerate(feature_names):
                # exclude feature
                Xtrain = X_train.loc[:, X_train.columns != f'{column}']
                Xval = X_validation.loc[:, X_validation.columns != f'{column}']
           
                # initialize and train model
                model = base_model(input_shape = (Nfeatures - 1,))
                history = model.fit(Xtrain, y_train, epochs=Nepochs, batch_size = batch_size, validation_data=(Xval, y_validation), verbose = 0)
                # store average loss results of last 10 epochs
                train_loss[n,i] = np.average(history.history['loss'][-10:])
                val_loss[n, i] = np.average(history.history['val_loss'][-10:])
            print(n+1," iterations out of ", Niterations, " completed")

        return train_loss, val_loss

def plot_loss_from_feature_dropping(train_loss_arr, val_loss_arr, categories):
    fig4, ax4 = plt.subplots()

    names = categories
    Niterations = train_loss_arr.shape[0]

    # calculate means and stds (errors on means)
    train_means = np.mean(train_loss_arr, axis = 0)
    val_means = np.mean(val_loss_arr, axis = 0) 

    train_stds = np.std(train_loss_arr, axis = 0, ddof = 1) / np.sqrt(Niterations)
    val_stds = np.std(val_loss_arr, axis = 0, ddof = 1) / np.sqrt(Niterations)

    ax4.errorbar(list(names), train_means[:-1], train_stds[:-1], fmt = 'ko', capthick = 1.5, capsize = 3, \
                elinewidth = 1.5,  markersize = 3, alpha = .7, label = 'Train loss')
    ax4.errorbar(list(names), val_means[:-1], val_stds[:-1], fmt = 'ro', capthick = 1.5, capsize = 3, \
                elinewidth = 1.5, markersize = 3, alpha = .7, label = 'Val. loss')
    ax4.plot(list(names), train_means[-1] * np.ones(len(list(names))), 'k-', label = 'Train loss baseline', lw = 1.5)
    ax4.plot(list(names), (train_means[-1] + train_stds[-1]) * np.ones(len(list(names))), 'k--', label = r'Train baseline SEM', lw=1)
    ax4.plot(list(names), (train_means[-1] - train_stds[-1]) * np.ones(len(list(names))), 'k--', lw =1)
    ax4.plot(list(names), (val_means[-1] + val_stds[-1]) * np.ones(len(list(names))), 'r--', label = r'Val. baseline SEM', lw=1)
    ax4.plot(list(names), (val_means[-1] - val_stds[-1]) * np.ones(len(list(names))), 'r--', lw =1)
    ax4.plot(list(names), val_means[-1] * np.ones((len(list(names)))), 'r-', label = 'Val. loss baseline', lw = 1.5)
    ax4.plot(['Levy'], 0.78e8, '.', markersize = 0.05)
    plt.xticks(fontsize = 9, rotation = -45)
    ax4.set(ylabel = r'$(y_{pred} - y_{true})^2$ loss', title = 'NN performance when excluding features')
    ax4.legend(fontsize = 10)

def evaluate_predictions(data, target_output, predicted_output, label = 'val. data'):
            rel_err = np.abs(predicted_output - target_output) / target_output
            print(f'For {label}:\n')
            for fraction in [10, 25, 50, 75, 95, 99]:
                print('Mean rel. error: ', rel_err.mean())
                print(f'{fraction} % of all predictions have a rel. error less than ', np.quantile(rel_err, fraction / 100))

def evaluate_predictions(y_pred, y_true, label = 'train', quantiles = None, verbose = 1):
            if verbose:
                print(f'\nFor {label} data:')

            f_list = [r2_score, metrics.mean_absolute_error, metrics.mean_absolute_percentage_error, mean_squared_log_error]
            name_list = ['r2', 'MAE', 'MAPE', 'MSQLE']
            res_list = [None] * len(f_list)
            for i, f in enumerate(f_list):
                if name_list[i] == 'MSQLE':
                    res_list[i] = f(y_true, np.abs(y_pred))
                else:
                    res_list[i] = f(y_true, y_pred)
                if verbose:
                    print(f'{name_list[i]}: ', res_list[i])

            rel_err = np.abs(y_pred - y_true) / y_true
            if quantiles is None:
                Nstatistics = 7
                quantiles = [.05, .25, .50, .75, .95, .99]
                statistics = [None] * Nstatistics
                statistics[0] = rel_err.mean()
                statistics[1:] = np.quantile(rel_err, quantiles)
            else:
                Nstatistics = len(quantiles)
                statistics = np.quantile(rel_err, quantiles)
            if verbose:
                print(f"mean and {quantiles} quantiles of rel error: ", statistics)

            return res_list, statistics
        

### MAIN ---------------------------------------------------------------------------------------

# Things to try:
# kfold cross validation

#PLOTS to produce
# rel error histogram after optimization. possibly unsigned (possible with true and predicted values)
# price distribution of real and predicted labels
## ANN performance baseline + ANN after losing 1st feature + ANN after.... 
## PRICE DIST FOR TEST; VAL; TRAINING
## NYT HEATMAP


## NEXT STEPS:
# ypred v ytrue and r^2. mean_abs_err, mean_abs_percent_err. + uncertainty 


# THEN do optimization of hyperparams(incldung grouping threshold, price_threshold)
# Use several metrics INCLUDING the quantile stuff.(use own metric)
# try upping complexety
# fig out what is going on with the line error


# THEN test it on test data. report price plot and quantile report for test and val data

# THEN do prediction analysis somehow. like: can we say something about when it is good or bad? 
# ...how does rel error (and abs error) change as a function of true price eg
# how does rel error change against predicted price? Can you use this to est. uncertainty on a prediction?
# ...uncertainty estimation?
# ...++search for other ides
# can you fig. out Why/How your very bad predictions come about?
# ...then write down all you have found! Or include randomforests and/or tress. But be mindful of the deadline


#WHEN TREES: extract features extraction

## Building and optimizing models
#--> at least 3, optimize acc to val data
# --> explain parameters and consider, loss functions, used metrics etc[MSE vs huber eller log-cosh]
# --> hyperparameter optimization
## --> also: model overshoots. Try capping top 1/2/5% of prices and see what happens? Also, maybe include the small prices
# -- try reducing threshold for grouping and see what happens - set as param.

# Feature selection: 
# Try using model to estimate importance, test performance after dropping each one and plot
# possibly do dimensionality reduction (PCA, VAE?)
# correlation?
# Drop features one by one and consider performance change

## Results
# --> av. rel. error, histogram of rel error
# --> histogram of real and predicted prices
# --> comparing methods
# --> combining methods and consider improvement
# --> prediction uncertainty estimation
# smirnov test of price dist


## if time:
#--> understanding if model performance varies across param values and use to create better model weighting


def main():
    ### PART 1: Load, clean and preprocess data
    data = pd.read_csv("train.csv", delimiter = ",") 
    data = data_cleaning(data, verbose = False)
   
    #visualize_data(data, numerical_plots=True, categorical_plots=True)

    

    
    # drop bad and irrelevant features
    drop_bad_features = True
    if drop_bad_features:
        features_to_drop = ['Doors', 'Wheel']
        #] ['Levy', 'Leather', 'Dr. Wheel', 'Manufac.', 'Airbags'] #['Cylinders', 'Manufac.', 'Model', 'Wheel'] #['Doors'] #, 'Levy'] #, 'Cylinders'] # 'Cylinders', 'Wheel',  'Color'] #, 'Levy', 'Model']
        #['Levy', 'Turbo', 'Airbags', 'Wheel', 'Doors', 'Cylinders', 'Leather', 'Eng. vol.', 'Mileage', 'Manufac.', 'Model', 'Color']
        #['Levy', 'Turbo', 'Airbags', 'Wheel', 'Doors', 'Cylinders']
        #['Levy', 'Turbo', 'Airbags', 'Wheel', 'Doors', 'Cylinders', 'Leather', 'Eng. vol.', 'Mileage', 'Manufac.', 'Model', 'Color']
        #['Doors', 'Wheel', 'Color', 'Airbags', 'Turbo', 'Cylinders', 'Mileage', 'Levy']  #['Model'] #
        data = data.drop(columns = features_to_drop)


    # Drop outliers in [Levy, Price, Mileage] according to IQR scheme
    print("No. of data points before dropping outliers: ", data.shape[0])
    drop_levy = False
    for feature in features_to_drop:
        if feature == 'Levy':
            drop_levy = True
    if drop_levy:
        outlier_features = ['Price', 'Mileage'] #,'Eng. vol.']
    else:
        outlier_features = ['Levy', 'Price', 'Mileage']
    data = remove_outliers(data, features=outlier_features, cutoff = 2.5)
    print("No. of data points after dropping outliers: ", data.shape[0])


    plot_prices = False
    if plot_prices:
        # Plot train and test distribution
        fig0, ax0 = plt.subplots()
        range = (0, data['Price'].max())
        bins = 160
        ax0.hist(data['Price'], bins = bins, range = range, \
                histtype='stepfilled', alpha=.3, density = True, edgecolor = 'black')
      
        ax0.set(xlabel = 'Price distribution', ylabel = 'Density', title = f'Price distribution of data')
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.ticklabel_format(style='sci', axis='x', scilimits=(1,4))
        ax0.legend()
     
        plt.show()


    # Choose how to encode categorical variables --- and normalize
    target_encoding, mixed_encoding = True, False
    plot_correlations = False
    if target_encoding:
        data = do_target_encoding(data, leave_one_out=True) #  # do_target_encoding(train_data, leave_one_out=True) #.select_dtypes(exclude='object') 
        _, Ncolumns = data.shape

        print("no of tot indices target encoding: ", Ncolumns)
        if plot_correlations:
            fig0, ax0 = plt.subplots(figsize=(6,12))
            ax0.set(title = 'Correlation heatmap using target encoding')
            sns.heatmap(data.corr(), square = True, annot = True, fmt='.2g') #, annot = True,) #, fmt='.1g')
        plt.show()
    elif mixed_encoding:
        data = do_mix_encoding(data)
        _, Ncolumns = data.shape 
        print("no of tot indices dummy encoding: ", Ncolumns)
 
        if plot_correlations:
            fig0, ax0 = plt.subplots(figsize=(6,12))
            ax0.set(title = 'Correlation heatmap using mixed encoding')
            sns.heatmap(data.corr(), square = True) #, annot = True,) #, fmt='.1g')
        plt.show()


    # Normalize
    data = rescale_input_linear(data, target_interval = [0,1])

    # Split into output and input variables
    output = data['Price'].astype('float')
    data = data.drop(columns = ['Price'])
    Ncolumns -= 1

    # Split ata into training and test data set
    X_training, X_test, y_training, y_test = train_test_split(data, output, test_size = 0.1, shuffle=True)
   
    # Split the training data set into a train and validation set
    X_train, X_val, y_train, y_val = train_test_split(X_training, y_training, test_size = 0.2, shuffle=True)

    N_train, N_val, N_test = X_train.shape[0], X_val.shape[0], X_test.shape[0]

    print("No. of train, val. and test data points: ", N_train, N_val, N_test)

  
    ### PART2: Build and optimize ML models
    run_neural_network, run_tree, run_many_models = True, True, True
    if run_neural_network:
       
        # Build neural network model
        def baseline_model(input_shape = (Ncolumns,), kernel_initializer = 'random_normal', learning_rate = 0.0007,  verbose = 1):
            l1 = False
            kwargs = {'kernel_initializer': kernel_initializer, 'activation': 'relu'}
            if l1:
                kwargs.update({'kernel_regularizer':regularizers.L1(0.0001)}), \
              #   'bias_regularizer':regularizers.L1(1e-5),
               #  'activity_regularizer':regularizers.L1(1e-5)})
            model = models.Sequential()
            
            model.add(layers.Dense(256, **kwargs, input_shape = input_shape))
            model.add(layers.Dense(512,**kwargs))
            model.add(layers.Dense(512, **kwargs))
            model.add(layers.Dense(512,**kwargs))

            model.add(layers.Dense(256, **kwargs))
            model.add(layers.Dense(256,**kwargs))
            model.add(layers.Dense(256, **kwargs))
            model.add(layers.Dense(256,**kwargs))  
            model.add(layers.Dense(256, **kwargs))
            model.add(layers.Dense(256,**kwargs))


            #model.add(layers.Dense(512,**kwargs))
            #model.add(layers.Dense(512,**kwargs))
            #model.add(layers.Dense(512,**kwargs))
            #model.add(layers.Dense(512,**kwargs))
     
            # kernel_regularizer = regularizers.L1L2(l1=1e-5,l2=1e-4)
            #model.add(layers.Dropout(0.3))

            model.add(layers.Dense(256, **kwargs))
            model.add(layers.Dense(256,**kwargs))
            model.add(layers.Dense(256, **kwargs)) 
            model.add(layers.Dense(256, **kwargs))
            model.add(layers.Dense(1, activation = 'linear')) #, activation = 'sigmoid'))
            model.build()

            if verbose:
                print(model.summary())
        
        # Compile
        # losses to try: cosh, huber. metrics to try: mean suqare error, mean absoulate percentage error, rootmeansquared error
        # params to play with: epochs, Nlayers, Nneurons, dropout, regularization, convolution, 
        #... batch size,
            metric =[tf.keras.metrics.RootMeanSquaredError()] #[tf.keras.metrics.MeanSquaredError()] #[tf.keras.metrics.LogCoshError()]# [tf.keras.metrics.LogCoshError()] #[tf.keras.metrics.MeanAbsolutePercentageError()] ## [tf.keras.metrics.MeanSquaredLogarithmicError()] # [tf.keras.metrics.MeanAbsolutePercentageError()], #
            loss = 'mse' #tf.keras.losses.LogCosh() #tf.keras.losses.MeanAbsolutePercentageError() # # tf.keras.losses.MeanSquaredLogarithmicError()
            kwargs_ann = {'optimizer': tf.keras.optimizers.Adam(learning_rate=learning_rate), \
                        'loss': loss, 'metrics': metric}   # 'loss': tf.keras.losses.BinaryCrossentropy(from_logits=True),
            model.compile(**kwargs_ann) 
            return model
    
        def baseline_model2():
            input_shape = (Ncolumns,)
            kernel_initializer = 'random_normal' 
            learning_rate = 0.0007  
            verbose = 1
            l1 = False
            kwargs = {'kernel_initializer': kernel_initializer, 'activation': 'relu'}
            if l1:
                kwargs.update({'kernel_regularizer':regularizers.L1(0.0001)}), \
              #   'bias_regularizer':regularizers.L1(1e-5),
               #  'activity_regularizer':regularizers.L1(1e-5)})
            model = models.Sequential()
            
            model.add(layers.Dense(256, **kwargs, input_shape = input_shape))
            model.add(layers.Dense(512,**kwargs))
            model.add(layers.Dense(512, **kwargs))
            model.add(layers.Dense(512,**kwargs))

            model.add(layers.Dense(256, **kwargs))
            model.add(layers.Dense(256,**kwargs))
            model.add(layers.Dense(256, **kwargs))
            model.add(layers.Dense(256,**kwargs))  
            model.add(layers.Dense(256, **kwargs))
            model.add(layers.Dense(256,**kwargs))


            #model.add(layers.Dense(512,**kwargs))
            #model.add(layers.Dense(512,**kwargs))
            #model.add(layers.Dense(512,**kwargs))
            #model.add(layers.Dense(512,**kwargs))
    
            # kernel_regularizer = regularizers.L1L2(l1=1e-5,l2=1e-4)
            #model.add(layers.Dropout(0.3))

            model.add(layers.Dense(256, **kwargs))
            model.add(layers.Dense(256,**kwargs))
            model.add(layers.Dense(256, **kwargs)) 
            model.add(layers.Dense(256, **kwargs))
            model.add(layers.Dense(1, activation = 'linear')) #, activation = 'sigmoid'))
            model.build()

            if verbose:
                print(model.summary())
        
        # Compile
        # losses to try: cosh, huber. metrics to try: mean suqare error, mean absoulate percentage error, rootmeansquared error
        # params to play with: epochs, Nlayers, Nneurons, dropout, regularization, convolution, 
        #... batch size,
            metric =[tf.keras.metrics.RootMeanSquaredError()] #[tf.keras.metrics.MeanSquaredError()] #[tf.keras.metrics.LogCoshError()]# [tf.keras.metrics.LogCoshError()] #[tf.keras.metrics.MeanAbsolutePercentageError()] ## [tf.keras.metrics.MeanSquaredLogarithmicError()] # [tf.keras.metrics.MeanAbsolutePercentageError()], #
            loss = 'mse' #tf.keras.losses.LogCosh() #tf.keras.losses.MeanAbsolutePercentageError() # # tf.keras.losses.MeanSquaredLogarithmicError()
            kwargs_ann = {'optimizer': tf.keras.optimizers.Adam(learning_rate=learning_rate), \
                        'loss': loss, 'metrics': metric}   # 'loss': tf.keras.losses.BinaryCrossentropy(from_logits=True),
            model.compile(**kwargs_ann) 
            return model
    
   
        # Estimate feature importance and discard bad and unimportant features
        # Decide whether to estimate feature importance, evaluate model performance after droppoing each feature, or simply run normally
        calc_feature_importance, evaluate_loss_from_feature_dropping, \
            drop_features, hyper_optimization, run_network, = False, False, False, False, True
        if calc_feature_importance:
            
            model = KerasRegressor(build_fn = baseline_model2, nb_epoch = 50, batch_size = 256, verbose=False)
            history = model.fit(X_train, y_train, epochs = 50, batch_size = 256, validation_data=(X_val, y_val))
          
            perm = PermutationImportance(model, n_iter = 10, cv = 'prefit').fit(X_train,y_train)
            fig2, ax2 = plt.subplots()
            plot_feature_importance(perm, X_train.columns, fig2, ax2)
            plt.show()
        elif evaluate_loss_from_feature_dropping:
            features_to_drop = ['Doors'] #, 'Cylinders'] #, 'Cylinders', 'Wheel', 'Dr. wheel', 'Color']
            categories = ['Levy', 'Category', 'Cylinders', 'Wheel', 'Color', 'Turbo']
            Xtrain = X_train.loc[:, ~X_train.columns.isin(features_to_drop)]
            Xval = X_val.loc[:, ~X_val.columns.isin(features_to_drop)]
            train_loss_arr, val_loss_arr = evaluate_feature_dropping(baseline_model, Xtrain, Xval, y_train, y_val,\
                                        Niterations = 2, Nepochs = 50, batch_size = 256, feature_names=categories)
            
            np.savetxt('train_loss_arr_v2_2.txt', train_loss_arr)
            np.savetxt('val_loss_arr_v2_2.txt', val_loss_arr)
            plot_loss_from_feature_dropping(train_loss_arr, val_loss_arr, categories = categories)
        elif drop_features:
            # Set which features to drop cumulatively in that order
            features_to_drop = ['Doors', 'Colors', 'Levy', 'Manufac', 'Cylinders']
                               # 'Cylinders', 'Wheel', 'Dr. wheel', 'Color'] # wheel, drive wheels, mileage
            colors = ['purple','teal', 'olivedrab', 'plum', 'magenta', 'plum', 'pink', 'green']
            Nepochs = 25
            Niterations = 2
            batch_size = 256
            Naverage = 5 # must be divisor of Nepochs
            Nrecordings = int(Nepochs / Naverage)
            train_loss_arr = np.zeros([Niterations, len(features_to_drop) + 1, Nrecordings])
            val_loss_arr = np.zeros_like(train_loss_arr)
            quantiles = [.10, .25, .50, .75, .95, .99]
            quantiles_arr = np.zeros([Niterations, len(features_to_drop) + 1, len(quantiles) + 1])
    
            fig1,ax1 = plt.subplots()
            iterations = np.arange(0,Nepochs, Naverage)
        
            for n in np.arange(Niterations):
                model = baseline_model()
                history = model.fit(X_train, y_train, epochs=Nepochs, batch_size = batch_size, validation_data=(X_val, y_val))

                # Make predictions
                y_val_pred = model.predict(X_val).flatten()
                rel_err = np.abs(y_val_pred - y_val) / y_val

                # record quantile stats
                quantiles_arr[n, -1, 0] = rel_err.mean()
                quantiles_arr[n, -1, 1:] = np.quantile(rel_err, quantiles)

                # record baseline losses
                train_loss_arr[n, -1,:] =  np.array(history.history['loss']).reshape(-1, Naverage).mean(axis = 1)
                val_loss_arr[n, -1,:] = np.array(history.history['val_loss']).reshape(-1,Naverage).mean(axis = 1)

            # calc and plot losses after dropping features
            name = ''
            names = ['Do', 'Co', 'Le', 'Ma', 'Cy']
            drop_features = []
            for i, column in enumerate(features_to_drop):
                print(i,"/",len(features_to_drop)," completed")
                drop_features.append(column)
                # exclude feature
                indices = X_train.columns.str.contains(column)
                Xtrain = X_train.loc[:, ~indices]
                Xval = X_val.loc[:, ~indices]
                No_features = Xval.shape[1]
                name = name + f' % {names[i]}'
                
                for n in np.arange(Niterations):
                    # initialize and train model
                    model = baseline_model(input_shape = (No_features,))
                    history = model.fit(Xtrain, y_train, epochs=Nepochs, batch_size = batch_size, validation_data=(Xval, y_val), verbose = 0)
                    
                    # Make predictions
                    y_val_pred = model.predict(Xval).flatten()
                    rel_err = np.abs(y_val_pred - y_val) / y_val

                    # record quantile stats
                    quantiles_arr[n, i, 0] = rel_err.mean()
                    quantiles_arr[n, i,1:] = np.quantile(rel_err, quantiles)

                    # store losses
                    train_loss_arr[n, i,:] = np.array(history.history['loss']).reshape(-1, Naverage).mean(axis = 1)
                    val_loss_arr[n, i,:] = np.array(history.history['val_loss']).reshape(-1, Naverage).mean(axis = 1)


            ## Calc mean, stds and save
            train_loss_arr_std = train_loss_arr.std(axis = 0, ddof = 1)
            train_loss_arr_mean = train_loss_arr.mean(axis = 0)
            train_loss_arr = np.r_['0', train_loss_arr_std[np.newaxis,:,:], train_loss_arr_mean[np.newaxis,:,:]]

            val_loss_arr_std = val_loss_arr.std(axis = 0, ddof = 1)
            val_loss_arr_mean = val_loss_arr.mean(axis = 0)
            val_loss_arr = np.r_['0', val_loss_arr_std[np.newaxis,:,:], val_loss_arr_mean[np.newaxis,:,:]]

            quantiles_arr_std = quantiles_arr.std(axis = 0, ddof = 1)
            quantiles_arr_mean = quantiles_arr.mean(axis = 0)
            quantiles_arr = np.r_['0', quantiles_arr_std[np.newaxis,:,:], quantiles_arr_mean[np.newaxis,:,:]]


            for i, name in enumerate(names):
                ax1.plot(iterations, train_loss_arr[1,i,:], '.-', color = f'{colors[i]}', label = f'TL {name}', markersize = 3, lw = 1.5)
                ax1.plot(iterations, val_loss_arr[1,i,:],'-', color = f'{colors[i]}', label = f'VL {name}', markersize = 3, lw =1.5)
                ax1.errorbar(iterations, train_loss_arr[1,i,:], train_loss_arr[0,i,:], fmt = '.', color = f'{colors[i]}', elinewidth=1, capsize=1,capthick=1)
                ax1.errorbar(iterations, val_loss_arr[1,i,:], val_loss_arr[0,i,:],fmt='.', color = f'{colors[i]}', elinewidth=1, capsize=1,capthick=1)
            
                print(f"For {name} model:\n")
                print("Mean rel. error: ", quantiles_arr[1, i, 0], "\u00B1", quantiles_arr[1, i, 0])
                for j, fraction in enumerate(quantiles):
                    print(f'{fraction*100} % of all predictions have a rel. error less than ', \
                        quantiles_arr[1, i, j+1], "\u00B1", quantiles_arr[0, i, j+1])


            ax1.plot(iterations, train_loss_arr[1,-1,:], 'k.-',  label = f'TL baseline', markersize = 3, lw = 1.5)
            ax1.plot(iterations, val_loss_arr[1,-1,:],'k-', label = f'VL baseline', markersize = 3, lw =1.5)
            ax1.errorbar(iterations, train_loss_arr[1,-1,:], train_loss_arr[0,-1,:], fmt = '.', color = f'{colors[i]}', elinewidth=1, capsize=1,capthick=1)
            ax1.errorbar(iterations, val_loss_arr[1,-1,:], val_loss_arr[0,-1,:],fmt='.', color = f'{colors[i]}', elinewidth=1, capsize=1,capthick=1)
            ax1.set(xlabel = 'Epochs', ylabel = r'$\ln \cosh(y_{pred} - y_{true})$ loss', title = 'NN loss when exluding features')
            ax1.legend(fontsize = 9)

            print("For baseline model:\n")
            print("Mean rel. error: ", quantiles_arr[1, -1, 0], "\u00B1", quantiles_arr[0, -1, 0])
            for i, fraction in enumerate(quantiles):
                print(f'{fraction*100} % of all predictions have a rel. error less than ', \
                    quantiles_arr[1, -1, i+1],"\u00B1", quantiles_arr[0, -1, i+1])

            # reshape and save
            train_loss_arr = train_loss_arr.reshape(-1, Nrecordings)
            val_loss_arr = val_loss_arr.reshape(-1, Nrecordings)
            quantiles_arr = quantiles_arr.reshape(-1, len(quantiles) + 1)

            np.savetxt('train_feature_drop0.txt', train_loss_arr)
            np.savetxt('val_feature_drop0.txt', val_loss_arr)
            np.savetxt('quantiles0.txt', quantiles_arr)       
        elif hyper_optimization:
            colors = ['purple','teal', 'olivedrab', 'plum', 'magenta', 'plum', 'pink', 'green']

            batch_sizes = [32, 64, 128]
            learning_rates = [0.0005, 0.001, 0.005, 0.01]
            ## ad hoc for epochs

            Nepochs = 100
            Nparams = len(batch_sizes) + len(learning_rates)
            Niterations = 5
            Naverage = 5 # must be divisor of Nepochs
            Nrecordings = int(Nepochs / Naverage)
            train_loss_arr = np.zeros([Niterations, Nparams, Nrecordings])
            val_loss_arr = np.zeros_like(train_loss_arr)
            quantiles = [.10, .25, .50, .75, .95, .99]
            quantiles_arr = np.zeros([Niterations, Nparams, len(quantiles) + 1])
    
            fig1,ax1 = plt.subplots()
            iterations = np.arange(0,Nepochs, Naverage)
        
            for i, column in enumerate(features_to_drop):
                print(i,"/",len(features_to_drop)," completed")
                drop_features.append(column)
                # exclude feature
                indices = X_train.columns.str.contains(column)
                Xtrain = X_train.loc[:, ~indices]
                Xval = X_val.loc[:, ~indices]
                No_features = Xval.shape[1]
                name = name + f' % {names[i]}'
                
                for n in np.arange(Niterations):
                    # initialize and train model
                    model = baseline_model(input_shape = (No_features,))
                    history = model.fit(Xtrain, y_train, epochs=Nepochs, batch_size = batch_size, validation_data=(Xval, y_val), verbose = 0)
                    
                    # Make predictions
                    y_val_pred = model.predict(Xval).flatten()
                    rel_err = np.abs(y_val_pred - y_val) / y_val

                    # record quantile stats
                    quantiles_arr[n, i, 0] = rel_err.mean()
                    quantiles_arr[n, i,1:] = np.quantile(rel_err, quantiles)

                    # store losses
                    train_loss_arr[n, i,:] = np.array(history.history['loss']).reshape(-1, Naverage).mean(axis = 1)
                    val_loss_arr[n, i,:] = np.array(history.history['val_loss']).reshape(-1, Naverage).mean(axis = 1)


            ## Calc mean, stds and save
            train_loss_arr_std = train_loss_arr.std(axis = 0, ddof = 1)
            train_loss_arr_mean = train_loss_arr.mean(axis = 0)
            train_loss_arr = np.r_['0', train_loss_arr_std[np.newaxis,:,:], train_loss_arr_mean[np.newaxis,:,:]]

            val_loss_arr_std = val_loss_arr.std(axis = 0, ddof = 1)
            val_loss_arr_mean = val_loss_arr.mean(axis = 0)
            val_loss_arr = np.r_['0', val_loss_arr_std[np.newaxis,:,:], val_loss_arr_mean[np.newaxis,:,:]]

            quantiles_arr_std = quantiles_arr.std(axis = 0, ddof = 1)
            quantiles_arr_mean = quantiles_arr.mean(axis = 0)
            quantiles_arr = np.r_['0', quantiles_arr_std[np.newaxis,:,:], quantiles_arr_mean[np.newaxis,:,:]]


            for i, name in enumerate(names):
                ax1.plot(iterations, train_loss_arr[1,i,:], '.-', color = f'{colors[i]}', label = f'TL {name}', markersize = 3, lw = 1.5)
                ax1.plot(iterations, val_loss_arr[1,i,:],'-', color = f'{colors[i]}', label = f'VL {name}', markersize = 3, lw =1.5)
                ax1.errorbar(iterations, train_loss_arr[1,i,:], train_loss_arr[0,i,:], fmt = '.', color = f'{colors[i]}', elinewidth=1, capsize=1,capthick=1)
                ax1.errorbar(iterations, val_loss_arr[1,i,:], val_loss_arr[0,i,:],fmt='.', color = f'{colors[i]}', elinewidth=1, capsize=1,capthick=1)
            
                print(f"For {name} model:\n")
                print("Mean rel. error: ", quantiles_arr[1, i, 0], "\u00B1", quantiles_arr[1, i, 0])
                for j, fraction in enumerate(quantiles):
                    print(f'{fraction*100} % of all predictions have a rel. error less than ', \
                        quantiles_arr[1, i, j+1], "\u00B1", quantiles_arr[0, i, j+1])


            ax1.plot(iterations, train_loss_arr[1,-1,:], 'k.-',  label = f'TL baseline', markersize = 3, lw = 1.5)
            ax1.plot(iterations, val_loss_arr[1,-1,:],'k-', label = f'VL baseline', markersize = 3, lw =1.5)
            ax1.errorbar(iterations, train_loss_arr[1,-1,:], train_loss_arr[0,-1,:], fmt = '.', color = f'{colors[i]}', elinewidth=1, capsize=1,capthick=1)
            ax1.errorbar(iterations, val_loss_arr[1,-1,:], val_loss_arr[0,-1,:],fmt='.', color = f'{colors[i]}', elinewidth=1, capsize=1,capthick=1)
            ax1.set(xlabel = 'Epochs', ylabel = r'$\ln \cosh(y_{pred} - y_{true})$ loss', title = 'NN loss when exluding features')
            ax1.legend(fontsize = 9)

            print("For baseline model:\n")
            print("Mean rel. error: ", quantiles_arr[1, -1, 0], "\u00B1", quantiles_arr[0, -1, 0])
            for i, fraction in enumerate(quantiles):
                print(f'{fraction*100} % of all predictions have a rel. error less than ', \
                    quantiles_arr[1, -1, i+1],"\u00B1", quantiles_arr[0, -1, i+1])

            # reshape and save
            train_loss_arr = train_loss_arr.reshape(-1, Nrecordings)
            val_loss_arr = val_loss_arr.reshape(-1, Nrecordings)
            quantiles_arr = quantiles_arr.reshape(-1, len(quantiles) + 1)

            np.savetxt('train_feature_drop.txt', train_loss_arr)
            np.savetxt('val_feature_drop.txt', val_loss_arr)
            np.savetxt('quantiles.txt', quantiles_arr)            
        elif run_neural_network:
            batch_size = 32 #1024
            Nepochs = 160
            Ntrials = 1
            Nstat = 4
            Nquant = 7
            stat = np.empty([Ntrials, Nstat, 3])
            quant = np.empty([Ntrials, Nquant, 3])
            train_loss_arr = np.empty([Ntrials, Nepochs])
            val_loss_arr = np.empty([Ntrials, Nepochs])
            for i in np.arange(Ntrials):
                model = baseline_model(input_shape = (Ncolumns,), \
                                    kernel_initializer='random_normal', learning_rate = 0.0007)
                history = model.fit(X_train, y_train, epochs=Nepochs, batch_size = batch_size, validation_data=(X_val, y_val))

                train_loss_arr[i,:] = history.history['loss']
                val_loss_arr[i,:] = history.history['val_loss']

      
                y_pred = model.predict(X_val).flatten()
                y_true_list = [y_train, y_val, y_test ]
                for j, X in enumerate([X_train, X_val, X_test]):
                    y_prediction = model.predict(X).flatten()
                    statistics, quantiles = evaluate_predictions(y_prediction, y_true_list[j])
                    stat[i,:, j] = statistics
                    quant[i,:,j] = quantiles


            # plot average losses
            fig6, ax6 = plt.subplots()
            loss = train_loss_arr.mean(axis = 0)
            loss_std = train_loss_arr.std(axis = 0, ddof = 1) / np.sqrt(Ntrials)
            val_loss = val_loss_arr.mean(axis = 0)
            val_loss_std = val_loss_arr.std(axis = 0, ddof = 1) / np.sqrt(Ntrials)
        
           # np.savetxt('tlm.txt',loss)
           # np.savetxt('tls.txt', loss_std)
           # np.savetxt('vlm.txt', val_loss)
           # np.savetxt('vls.txt',val_loss_std)

            iterations = np.arange(len(loss))

            ax6.set(xlabel = 'Epochs', ylabel = r'$(y_{pred} - y_{true})^2$ loss', title = 'Neural network training')
            ax6.plot(iterations, loss, '-', label = 'Train loss', color = 'orange', alpha =.8)
            ax6.errorbar(iterations, loss, loss_std, fmt = 'k.', elinewidth = 1, capsize = 1, capthick = 1, )
            ax6.plot(iterations, val_loss,'-', label = 'Validation loss', color = 'olivedrab')
            ax6.errorbar(iterations, val_loss, val_loss_std, fmt = 'k.', elinewidth = 1, capsize = 1, capthick = 1, )
            ax6.legend()
            
            # print results
            names = ['Train', 'Val', 'Test']
            print("Rel err mean and quantiles: ")
            print(quant.mean(axis=0))
            print(quant.std(axis = 0, ddof = 1) / np.sqrt(Ntrials))
            print("\nStatistics:")
            print(stat.mean(axis = 0))
            print(stat.std(axis = 0, ddof = 1) / np.sqrt(Ntrials))
          
            y_pred_test = model.predict(X_test).flatten()

            _,cnn_quant = evaluate_predictions(y_pred_test,y_test, quantiles = np.arange(0,1,0.01))
            print(cnn_quant)

            res, _ = evaluate_predictions(y_pred, y_val, label = 'val')
            r2_test = r2_score(y_pred_test, y_test)
            ## Plot predicted price vs true price
            fig11, ax11 = plt.subplots(ncols = 2)
            ax11 = ax11.flatten()
            
            ax11[0].plot(y_val, y_pred, '.', markersize = 2.5, alpha = .4, label = 'Val. data')
            ax11[1].plot(y_test, y_pred_test, '.', markersize = 2.5, color='navy', alpha = .4, label = 'Test data')
            ax11[0].set_ylim((0,60_000))
            ax11[1].set_ylim((0,60_000))
            
            for ax in ax11.flat:
                ax.set(xlabel = 'True price', ylabel = 'Predicted price')
                ax.label_outer()

            fig11.suptitle('Predicted vs. true price for test and val. data', fontsize = 14)
           # fig11.supxlabel('True Price', fontsize = 13)
           # fig11.supylabel('Predicted Price', fontsize = 13)
           
            ax11[0].legend()
            ax11[1].legend()
            fig11.tight_layout()
            ax11[0].text(3000, 50_000, fr'$R^2$={res[0]:.2f}', fontsize=12)
            ax11[1].text(3000, 50_000, fr'$R^2$={r2_test:.2f}', fontsize=12)
      
           # evaluate_predictions(X_val, y_val, model.predict(X_val).flatten(), label = 'val. data')
        
    # Part 3: Build and optimize tree classifier
    if run_tree:

        ## Decide which scheme to run
        GB, ADA = True, False
        if GB:
            ## Set Graident boosting parameters
            kwargs_BDT = {'loss': "squared_error", 'n_estimators': 15, 'learning_rate': 1, 'max_depth': 2, 'criterion': "friedman_mse", \
                            'min_samples_leaf': 20, 'min_samples_split': 20, 'subsample': .7, 'max_features': None,\
                                'n_iter_no_change': None, 'validation_fraction': 0.1, 'tol': 0.0001, 'ccp_alpha':  0.3, \
                                    'min_impurity_decrease': .2, 'min_weight_fraction_leaf': 0.0001}
            bdt = GradientBoostingRegressor(**kwargs_BDT)
        elif ADA: #TOPLAYWITH: loss func, normal params, pruning ccp, automatic stopping thingy? min decrease?
            ## Set AdaBoost parameters
            kwargs_BDT = {'n_estimators': 100, 'learning_rate': 1., 'loss':'linear',} # default are 50,1
            ## Set decision tree parameters
            kwargs_DT = { 'max_depth': 8, 'criterion': "squared_error", 'min_samples_split': 15, 'min_samples_leaf': 25, \
                        'splitter': "best", 'max_features': 10, 'ccp_alpha':  0.3, 'min_impurity_decrease': 0.1, \
                            'min_weight_fraction_leaf': 0.0001} 
            # default are 1, gini, 2, 1, best, 0. For all parameters, use BDT.get_params()    
            bdt = AdaBoostRegressor(DecisionTreeRegressor(**kwargs_DT), **kwargs_BDT)
        else:
            bdt = DecisionTreeRegressor()
        search_best_parameters, calc_permutation_importance = False, False
        if search_best_parameters:
            ## define parameters and ranges to vary
            param_grid = {'n_estimators': list(np.arange(24,25,2).astype('int')),\
                        'base_estimator__max_depth': list(np.arange(2,3,2).astype('int')),\
                            'learning_rate': list(np.arange(0.2,0.3,0.2)), \
                                'base_estimator__max_features': list(np.arange(14,16))}
        
            bdt = AdaBoostRegressor(DecisionTreeRegressor(**kwargs_DT), **kwargs_BDT)

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

       # bdt = DecisionTreeRegressor()
        bdt.fit(X_train, y_train)

        if calc_permutation_importance:
            clf = bdt.fit(X_train, y_train) 
            result = permutation_importance(clf, X_train, y_train, n_repeats=10,random_state=0)
            print(result.importances_mean)
            print(result.importances_std)

       # print(X_train.columns)
       # print(bdt.feature_importances_)
        if 0:
            train_generator = bdt.staged_predict(X_train)
            val_generator = bdt.staged_predict(X_val)
            train_loss = []
            val_loss = []
            for val in train_generator:
                y_pred_train = val
                loss = np.sqrt(((y_pred_train - y_train) ** 2 ).mean())
                train_loss.append(loss)
            for val in val_generator:
                y_pred_val = val
                loss = np.sqrt(((y_pred_val - y_val) ** 2).mean())
                val_loss.append(loss)


            iterations = np.arange(len(val_loss))
            fig9, ax9 = plt.subplots()

            ax9.set(xlabel = 'Epochs', ylabel = r'$\ln \cosh(y_{pred} - y_{true})$ loss', title = 'AdaBoost traning and validation loss')
            ax9.plot(iterations, train_loss, 'k.', label = 'Train loss', markersize = 3)
            ax9.plot(iterations, val_loss,'r.', label = 'Validation loss', markersize = 3)
            ax9.legend()
           
        if not run_many_models:
            y_pred_train = bdt.predict(X_train)
            y_pred_val = bdt.predict(X_val)
            
            evaluate_predictions(y_pred_train, y_train)
            evaluate_predictions(y_pred_val, y_val)
    # Part 4: Build and run many models
    if run_many_models:

        

        CV = []
        R2_train = []
        R2_val = []
        R2_test = []

        def car_pred_model(ml_model):
            stat_arr = np.empty(4)
            quant = np.empty(100)
            quantiles = np.arange(0,1,0.01)
          
            model = ml_model
            # Training model
            model.fit(X_train,y_train)
            
            if 0:   
                # R2 score of train set
                y_pred_train = model.predict(X_train)
                R2_train_model = r2_score(y_train,y_pred_train)
                R2_train.append(round(R2_train_model,2))
                stat_arr[:,0], quant[:,0] = evaluate_predictions(y_pred_train, y_train, quantiles = np.arange(0,1,0.01), verbose = False)

                # R2 score of val. set
                y_pred_val = model.predict(X_val)
                R2_val_model = r2_score(y_val,y_pred_val)
                R2_val.append(round(R2_val_model,2))
                stat_arr[:,1], quant[:,1] = evaluate_predictions(y_pred_val, y_val, quantiles = np.arange(0,1,0.01), label = 'val', verbose = False)
            
            # R2 score of test set
            y_pred_test = model.predict(X_test)
            R2_test_model = r2_score(y_test,y_pred_test)
            R2_test.append(round(R2_test_model,2))
            stat_arr[:], quant[:] = evaluate_predictions(y_pred_test, y_test, quantiles = np.arange(0,1,0.01), label = 'test', verbose= False)
                
            # Printing results
         #   print("Train R2-score :",round(R2_train_model,2))
          #  print("Val R2_score", round(R2_train_model,2))
            print("Test R2-score :",round(R2_test_model,2))
          
            if 0:
                # Plotting Graphs 
                # Residual Plot of train data
                fig, ax = plt.subplots(1,2,figsize = (10,4))
                ax[0].set_title('Residual Plot of Train samples')
                sns.distplot((y_train-y_pred_train),hist = False,ax = ax[0])
                ax[0].set_xlabel('y_train - y_pred_train')
                
                # Y_test vs Y_train scatter plot
                ax[1].set_title('y_test vs y_pred_test')
                ax[1].scatter(x = y_test, y = y_pred_test)
                ax[1].set_xlabel('y_test')
                ax[1].set_ylabel('y_pred_test')
                
        
            return stat_arr, quant




        # XG boost
        bdt = GradientBoostingRegressor(**kwargs_BDT)
        # linear
        lr = LinearRegression()
        # ridge
        ridge_kwargs = dict(alpha = 0.3263)
        rg = Ridge(**ridge_kwargs)
        #ridge_best = GridSearchCV(estimator = rg, param_grid = ridge_kwargs, cv = 5, n_jobs = -1).fit(X_train, y_train)
        #print(ridge_best.best_params_)
        # Lasso
        ls_kwargs = dict(alpha = 0.7448)
        ls = Lasso(**ls_kwargs)
    #    ls_best = GridSearchCV(estimator = ls, param_grid = ls_kwargs, cv = 5, n_jobs = -1).fit(X_train, y_train)
     #   print(ls_best.best_params_)
        # knn
        knn_kwargs = dict(n_neighbors = 7, weights = 'distance', p = 1)
        knn = KNeighborsRegressor(**knn_kwargs)
        
        #knn_best = GridSearchCV(estimator = knn, param_grid = knn_kwargs, cv = 5, n_jobs = -1).fit(X_train, y_train)
        #print(knn_best.best_params_)
        # Random forest
        rf_kwargs = dict(n_estimators = 500, min_samples_split = 20, min_samples_leaf = 20, max_depth = 4)
        rf = RandomForestRegressor(**rf_kwargs)

        if 0:
            # Number of trees in Random forest
            n_estimators=list(np.arange(350,650,50))
            # Maximum number of levels in a tree
            max_depth=list(np.arange(6,10))
            # Minimum number of samples required to split an internal node
            min_samples_split=list(np.arange(1,5))
            # Minimum number of samples required to be at a leaf node.
        ## min_samples_leaf=[1,2,5,7]
            # Number of fearures to be considered at each split
        # max_features=['auto','sqrt']

            # Hyperparameters dict
            param_grid = {"n_estimators":n_estimators,
                        "max_depth":max_depth,
                        "min_samples_split":min_samples_split}
                    # "min_samples_leaf":min_samples_leaf},
                    # "max_features":max_features}

            rf_best = RandomizedSearchCV(estimator = rf, param_distributions = param_grid, cv = 5, n_jobs = -1).fit(X_train, y_train)
            print(rf_best.best_params_)


        Ntrials = 5
        Nmodels = 5
        stat = np.empty([Ntrials, 4, Nmodels])
        quant = np.empty([Ntrials, 100, Nmodels])
        model_list = [lr, ls, knn, rf, bdt]


        for i in np.arange(Ntrials):
            for j, model in enumerate(model_list):
                stat[i,:,j], quant[i,:,j] = car_pred_model(model)
                
           
        print(stat.mean(axis=0))
        print(stat.std(axis=0,ddof=1)/np.sqrt(Ntrials))

     #   np.savetxt('q4.txt',quant[0,:,:])
        for i in np.arange(Ntrials):
            name = f'q{i}.txt'
        quant[i,:,:3] = np.loadtxt(name)

     
        fig13, ax13 = plt.subplots()
        ax13.set(xlabel = 'Quantile', ylabel = 'Relative error', title = 'Rel. error against quantile for different models')
        quantiles = np.linspace(0,1,100)
       
        model_names = ['Linear', 'Lasso', 'KNN', 'Random forest', 'XG Boost']
        re = quant.mean(axis = 0)
        std = quant.std(axis = 0, ddof = 1) / np.sqrt(Ntrials)
        ax13.plot(quantiles, cnn_quant, label = 'ANN')
        for i, model in enumerate(model_names):
            ax13.plot(quantiles, re[:,i], label = f'{model_names[i]}')
          #  ax13.errorbar(quantiles, re[:,i], std[:,i], fmt = 'k.', markersize = 1)
        ax13.legend()


    # FINAL PART: Plot and analyse results

    do_price_plots, visualize_feature_drop_loss, = False, False
    plot_test = False
 
    if run_tree:

        model = bdt
    if visualize_feature_drop_loss:
    # Set which features to drop cumulatively in that order
        Nepochs = 350
        Naverage = 5
        Niterations = 2
        batch_size = 128

        Nrecordings = int(Nepochs / Naverage)
        quantiles = [.10, .25, .50, .75, .95, .99]
        colors = ['purple', 'teal', 'orange', 'red', 'green', 'navy']
       
        train_loss_arr = np.loadtxt('train_feature_drop.txt').reshape(Niterations, - 1, Nrecordings)
        val_loss_arr = np.loadtxt('val_feature_drop.txt').reshape(Niterations, - 1, Nrecordings)

        fig1,ax1 = plt.subplots()
        iterations = np.arange(0,Nepochs, Naverage)
    
        # calc and plot losses after dropping features
        name = ''
        names = ['Do', 'Cy', 'Wh', 'Dr', 'Co']
        drop_features = []
        
        for i, nam in enumerate(names):
            if i == 0:
                name = nam
            else:
                name = name + '+' + nam
            iterations = iterations + .5* i
            ax1.plot(iterations, train_loss_arr[1,i,:], '.-', color = f'{colors[i]}', markersize = 1, lw = 1.5, alpha =.6)
            ax1.plot(iterations, val_loss_arr[1,i,:],'-', color = f'{colors[i]}', label = f'{name}', markersize = 1, lw =1.5, alpha =.6)
          #  ax1.errorbar(iterations, train_loss_arr[1,i,:], train_loss_arr[0,i,:], fmt = '.', markersize=1, color = f'{colors[i]}', elinewidth=1, capsize=1,capthick=1, alpha =.6)
         #   ax1.errorbar(iterations, val_loss_arr[1,i,:], val_loss_arr[0,i,:],fmt='.', markersize=1,color = f'{colors[i]}', elinewidth=1, capsize=1,capthick=1, alpha =.6)
        
        iterations = iterations + .5
        ax1.plot(iterations, train_loss_arr[1,-1,:], 'k.-', label = f'Baseline',  markersize = 1, lw = 1.5)
        ax1.plot(iterations, val_loss_arr[1,-1,:],'k-', markersize = 1, lw =1.5)
        ax1.errorbar(iterations, train_loss_arr[1,-1,:], train_loss_arr[0,-1,:], markersize=1, fmt = 'k.',elinewidth=1, capsize=1,capthick=1, label = f'Baseline uncertainty')
        ax1.errorbar(iterations, val_loss_arr[1,-1,:], val_loss_arr[0,-1,:],fmt='k.',  markersize=1, elinewidth=1, capsize=1,capthick=1)
        ax1.set(xlabel = 'Epochs', ylabel = r'$\ln \cosh(y_{pred} - y_{true})$ loss', title = 'NN loss when exluding features')
        ax1.legend(loc = 'upper left', fontsize = 10,)
        ax1.annotate('Train losses', xy=(200, 3000), xytext=(250, 3500),
            arrowprops=dict(facecolor='black', shrink=0.02), fontsize = 11)
        ax1.annotate('Validation losses', xy=(200, 5000), xytext=(250, 5500),
            arrowprops=dict(facecolor='black', shrink=0.02), fontsize = 11)

    if do_price_plots:
        if plot_test:
            data_list = [X_val, X_test]
        else:
            data_list = [X_val]

        target_list = [y_val, y_test]
        name_list = ['val. data', 'test data']
        histtypes_list = ['step', 'stepfilled']

      #  fig3, ax3 = plt.subplots()
       # fig5, ax5 = plt.subplots()
        #fig7,ax7 = plt.subplots()
        #fig8, ax8 = plt.subplots()
        fig10, ax10 = plt.subplots()

        for i, data in enumerate(data_list):
            y_pred = model.predict(data).flatten()

            evaluate_predictions(y_pred, target_list[i], label = name_list[i])

            rel_err = np.abs(y_pred - target_list[i]) / target_list[i]
            rel_err_signed = (y_pred - target_list[i]) / target_list[i]

         
            ## Plot predicted price vs true price
            ax10.plot(target_list[i], y_pred, '.', markersize = 1.5, alpha = .6)
            ax10.set(xlabel = 'True price', ylabel = 'Predicted price', title = 'Predicted vs. true price')

            if 0:
                # Plot rel error against true price
                ax7.scatter(target_list[i], rel_err_signed, color = 'red', alpha = .5, label = 'Signed rel. error', s = 3)
                #ax7.scatter(target_list[i], rel_err, color = 'black', alpha = .5, label = 'Unsigned rel. error', s = 3)
                ax7.set(xlabel = 'True price', ylabel = 'Relative error', title = 'Relative error against true price')
                ax7.legend()

                # Plot rel error against predicted price
                ax8.scatter(y_pred, rel_err_signed, color = 'red', alpha = .5, label = 'Signed rel. error', s = 3)
            # ax8.scatter(y_pred, rel_err, color = 'black', alpha = .5, label = 'Unsigned rel. error', s = 3)
                ax8.set(xlabel = 'Predicted price', ylabel = 'Relative error', title = 'Relative error against true price')
                ax8.legend()

                # Plot rel. prediction errors
            #  ax3.hist(rel_err, range=(-5,5),bins=60, color = 'black', label ='Unsigned', alpha = .5)
                ax3.hist(rel_err_signed, range=(-5,5),bins=60, color = 'red', label ='Signed', alpha = .5)
                ax3.set(xlabel = r'$(Price_{predicted} - Price_{true})/Price_{true}$', ylabel = 'Count', title = 'Signed relative prediction error')
                ax3.legend()
                # plot true and predicted price histograms
                range = (0, 80_000)
                bins = 80
                res_smirnov = stats.ks_2samp(y_pred, target_list[i])
                d = {'Smirnov p-value': res_smirnov[1]}
                text = nice_string_output(d, extra_spacing=2, decimals=2)
                add_text_to_ax(0.50, 0.76, text, ax5, fontsize=13)
                print("Smirnov test statistic and p value: ", res_smirnov[0], res_smirnov[1])
                ax5.hist(target_list[i], range = range, bins = bins, histtype = f'{histtypes_list[i]}'\
                        ,color = 'teal', alpha = .6, label = f'True price distribution for {name_list[i]}', linewidth = 1.5)
                ax5.hist(y_pred, range = range, histtype = f'{histtypes_list[i]}', bins = bins, \
                        color = 'plum', alpha = .6, label = f'Predicted price distribution {name_list[i]}', linewidth = 1.5)
                ax5.set(xlabel = 'Price', ylabel = 'Count', title = f'True and prediction price dist. for {name_list[i]}')
                ax5.legend()
                fig5.tight_layout()

    plt.show()

if __name__ == '__main__':
    main()

