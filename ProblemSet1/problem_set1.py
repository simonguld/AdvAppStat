# Author: Simon Guldager Andersen
# Date (latest update): 15-02-2023

### SETUP ------------------------------------------------------------------------------------

## Imports:
import os
import numpy as np
import pandas as pd
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from cycler import cycler


## Change directory to current one
#os.chdir('AdvAppStat\ProblemSet1')

# Decide whether to include BE in the list of conferences
include_BE = True


### FUNCTIONS ----------------------------------------------------------------------------------

## The following functions are written and provided by Troels C. Petersen in the course 'Applied Statistics'. 
# They serve as an easy way to add data entries to plots

def format_value(value, decimals):
    """ 
    Checks the type of a variable and formats it accordingly.
    Floats has 'decimals' number of decimals.
    """
    
    if isinstance(value, (float, np.float64)):
        return f'{value:.{decimals}f}'
    elif isinstance(value, (int, np.integer)):
        return f'{value:d}'
    else:
        return f'{value}'

def values_to_string(values, decimals):
    """ 
    Loops over all elements of 'values' and returns list of strings
    with proper formating according to the function 'format_value'. 
    """
    
    res = []
    for value in values:
        if isinstance(value, list):
            tmp = [format_value(val, decimals) for val in value]
            res.append(f'{tmp[0]} +/- {tmp[1]}')
        else:
            res.append(format_value(value, decimals))
    return res

def len_of_longest_string(s):
    """ Returns the length of the longest string in a list of strings """
    return len(max(s, key=len))

def nice_string_output(d, extra_spacing=5, decimals=3):
    """ 
    Takes a dictionary d consisting of names and values to be properly formatted.
    Makes sure that the distance between the names and the values in the printed
    output has a minimum distance of 'extra_spacing'. One can change the number
    of decimals using the 'decimals' keyword.  
    """
    
    names = d.keys()
    max_names = len_of_longest_string(names)
    
    values = values_to_string(d.values(), decimals=decimals)
    max_values = len_of_longest_string(values)
    
    string = ""
    for name, value in zip(names, values):
        spacing = extra_spacing + max_values + max_names - len(name) - 1 
        string += "{name:s} {value:>{spacing}} \n".format(name=name, value=value, spacing=spacing)
    return string[:-2]

def add_text_to_ax(x_coord, y_coord, string, ax, fontsize=12, color='k'):
    """ Shortcut to add text to an ax with proper font. Relative coords."""
    ax.text(x_coord, y_coord, string, family='monospace', fontsize=fontsize,
            transform=ax.transAxes, verticalalignment='top', color=color)
    return None


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
rcParams['axes.prop_cycle'] = cycler(color = ['teal', 'navy', 'coral', 'plum', 'red', \
         'black', 'olivedrab','purple', 'cyan', 'yellow', 'khaki','lightblue'])
np.set_printoptions(precision = 5, suppress=1e-10)
pd.options.mode.chained_assignment = None

def main():

    # Construct list of conferences of interest
    conferences = ['ACC', 'SEC', 'B10', 'BSky', 'A10']
    if include_BE:
        conferences.append('BE')  
    
    print("The conferences considered in the following are: ", conferences)
    print("\n\n")



    ## Load data
    data2009 = pd.read_excel('Data2009.xlsx')
    data2014 = pd.read_excel("Data2014.xlsx")

    data_list = [data2009, data2014]

    for k, data in enumerate(data_list):
        ## Column names reccur several times in the file. Below we remove them
        redundant_rows = data.index[data['AdjEM'] == 'AdjEM']
        redundant_rows2 = redundant_rows - 1
        redundant_rows = np.r_['0', redundant_rows, redundant_rows2]
        data = data.drop(redundant_rows)

        ## Rename unnamed columns (which represent the rank of the given team in terms of the previous variable)
        for i,col in enumerate(data.columns):
            if col == f'Unnamed: {i}':
                name = data.columns[i-1] + '_rank'
        
                data.rename(index = {f'{i}':f'{i}'}, columns = {f'{col}':f'{name}'}, inplace = True)

        ## Convert numeric entries to float
        str_columns = ['Team', 'Conf', 'W-L']
        for col in data.columns:
            str_column = False
            for str_name in str_columns:
                if col == str_name:
                    str_column = True
            if str_column == False:
                data[col] = pd.to_numeric(data[col])

        ## Remove numbers following team names
        numbers = np.arange(10)
        for i, name in enumerate(data['Team']):
            bad_format = True
            while bad_format:
                bad_format = False
                for num in numbers:
                    if name[-1] == str(num): # or name[-1] == " ":
                        name = name[:-1]
                        bad_format = True
            name = name.strip()
          
            data['Team'].iloc[i] = name

        data_list[k] = data
   
    data2009 = data_list[0]
    data2014 = data_list[1]

    print("No. of teams in 2009 and 2014, respectively: ", len(data2009['Team'])," ",len(data2014['Team']))



    ## EXC. 1:
    # Take the 2014 data, produce histograms of AdjD for all teams in the following 5 conferences

    Nteams = 0
    # Extract relevant teams and create histogram
    fig1, ax1 = plt.subplots()
    range = (85,115)
    bins = int ((range[1] - range[0]) / 1)
    ax1.set(xlabel = 'Adjusted Defense (AdjD)', ylabel = 'Count', title = 'AdjD for all teams attending each conference')

    for i, conf in enumerate(conferences):
        indices = data2014.index[data2014['Conf'] == conf]
        data = data2014['AdjD'][indices]
        Nteams += len(indices)

        if conf == 'B10' or conf == 'ACC':
            histtype = 'stepfilled'
            opauge = .4
        elif conf == 'BE':
            opauge = .8
        else:
            histtype = 'step'
            opauge = .5
        ax1.hist(data, bins = bins, range = range, lw = 2, histtype=histtype, alpha = opauge, label = f'{conf}')

    text = nice_string_output({'Entries': Nteams}, extra_spacing=2, decimals=3)
    add_text_to_ax(0.25, 0.9, text, ax1, fontsize=16)

    fig1.tight_layout()
    ax1.legend(loc = 'upper left')
  


    ## EXC 2:
    # Consider 2009 and 2014 data, and calc. the diff. in Adj0 (2014-2009) for all teams in the 5 conferences.
    # Plot them, same colors as above
    fig2, ax2 = plt.subplots()
    ax2.set(xlabel = '2009 AdjO value', ylabel = 'AdjO difference (2014-2009)', title = 'AdjO diff. against 2009 AdjO value')  
    Nteams_both_years = 0

    # Save the teams that played the conferences in 09 but not in 14
    conf_teams_not14 = []

    for conf in conferences:
        indices09 = data2009.index[data2009['Conf'] == conf]
        AdjO_09_tot = data2009['AdjO'][indices09]
        teams_09 = data2009['Team'][indices09]


        ## Extract correspondding values for AdjO-14:        
        AdjO_09 = []
        AdjO_14 = []
        for i, team in enumerate(teams_09):
        
            index14 = data2014.index[data2014['Team'] == team]

            ## Include only if team went to same conference in 2014
            if data2014['Conf'][int(np.array([index14]))] == conf:
                AdjO_14.append(np.float64(data2014['AdjO'][index14]))
                AdjO_09.append(np.float64(AdjO_09_tot.iloc[i]))
                Nteams_both_years += 1
            else:
                print(conf, " ", team, " with index ", int(np.array(index14)), " were not in 2014 conference")
                conf_teams_not14.append(team)

        # Calculate difference
        AdjO_14 = pd.Series(AdjO_14)
        AdjO_09 = pd.Series(AdjO_09)
        AdjO_diff = AdjO_14 - AdjO_09
        std = AdjO_diff.std(ddof = 1)
        SEM = std/np.sqrt(len(AdjO_diff))

        # Calculate mean and median
        print(f'AdjO diff. mean and median for {conf}: ', np.round(AdjO_diff.mean(),3), "\u00B1", np.round(SEM,3), "   ",np.round(AdjO_diff.median(),9))

        # Plot data
        ax2.plot(AdjO_09, AdjO_diff, '.', markersize = '12',label = f'{conf}', alpha = .5)

        text = nice_string_output({'Entries': Nteams_both_years}, extra_spacing=2, decimals=3)
        add_text_to_ax(0.05, 0.45, text, ax2, fontsize=16)
        fig2.tight_layout()
        ax2.legend(loc = 'lower left')


    ### Calc diff in AdjO for all teams with data in both years.
    ### Calc median and mean for all teams; For each of the 5 conferences; for all teams not in conferences 
 
    # Step one: Find all teams present in both years
    AdjO_09_both_years = []
    AdjO_14_both_years = []

    names_both_years = []
    conf_both_years = []
    indices_09_both_years = []
    print("\n")
    for i, team in enumerate(data2009['Team']):
        index14 = data2014.index[data2014['Team'] == team]
        # If team is also present in 2014 data, add the data to the lists
        if len(index14) != 0:
            names_both_years.append(team)
            conf_both_years.append(data2009['Conf'].iloc[i])
            indices_09_both_years.append(i)
            AdjO_09_both_years.append(np.float64(data2009['AdjO'].iloc[i]))

            index14_both_years = data2014.index[data2014['Team'] == team]
            AdjO_14_both_years.append(np.float64(data2014['AdjO'][index14_both_years]))
        else:
            print(team, " is present in 2009 but not in 2014" )

 
    # Verify equal lengths and convert to series
    assert(len(AdjO_09_both_years) == len(AdjO_14_both_years))
    print("\nNo. of teams present both years: ", len(AdjO_09_both_years))

    names_both_years = pd.Series(names_both_years)
    conf_both_years = pd.Series(conf_both_years)
    indices_09_both_years = pd.Series(indices_09_both_years)

    AdjO_09_both_years = pd.Series(AdjO_09_both_years)
    AdjO_14_both_years = pd.Series(AdjO_14_both_years)

   

    # Calculate mean and median for Adjo diff.
    AdjO_diff_tot = AdjO_14_both_years - AdjO_09_both_years
    std = AdjO_diff_tot.std(ddof = 1)
    SEM = std/np.sqrt(len(AdjO_diff_tot))
    print("Mean and median for all teams with data in both years: ", np.round(AdjO_diff_tot.mean(), 3), "\u00B1", np.round(SEM,3), "   ", np.round(AdjO_diff_tot.median(),9))


    ## Calc. mean and median for all teams not in conference
    # Construct mask for teams with data in 2009 and 2014 attending one of the conferences in 2009
    for j, conf in enumerate(conferences):
        indices14 = data2014.index[data2014['Conf'] == conf]
        names14 = data2014['Team'][indices14]
        if j == 0:
            mask = (conf_both_years == conf)
        else:
            mask = ((mask) | (conf_both_years == conf))

        ## Include teams present in both years attending conf. in 2014 but not in 2009
        for i, name in enumerate(names14):
            name_index = names_both_years.index[names_both_years == name]
            if len(name_index) !=0:
                mask[name_index] = True

    print("\nNo. of teams, with data in 09 and 14, that didn't attend either conference either year: ", len(AdjO_14_both_years[~mask]))
    print("No. of teams, with data in 09 and 14, that attended one of the conferences at least one year", len(AdjO_14_both_years[mask]))

    AdjO_diff_not_conf = AdjO_14_both_years[~mask] - AdjO_09_both_years[~mask]
    std = AdjO_diff_not_conf.std(ddof = 1)
    SEM = std/np.sqrt(len(AdjO_diff_not_conf))
    print("Mean and median for all teams with data in both years not in conferences: "\
        , np.round(AdjO_diff_not_conf.mean(), 3), "\u00B1", np.round(SEM,3), "   ", np.round(AdjO_diff_not_conf.median(),9))

    plt.show()

if __name__ == '__main__':
    main()
