'''
Data Analysis and Interpretation Capstone
The Association between GDP Per Capita and Various Development Indicators
Analysis of World Bank data from 2012, with verification on 2013 data
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv('worldbank.csv', low_memory=False)


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
# Suppress warnings for chained indexing
pd.options.mode.chained_assignment = None

print ("Number of observations:", len(data)) #number of observations (rows)
print ("Number of variables:", len(data.columns)) # number of variables (columns)
print ('')

# Codebook
'''
Variable	Label
x142_2012	GDP PER CAPITA (CURRENT US$)
x126_2012	FIXED BROADBAND SUBSCRIPTIONS (PER 100 PEOPLE)
x156_2012	IMPROVED WATER SOURCE (% OF POPULATION WITH ACCESS)
x167_2012	INTERNET USERS (PER 100 PEOPLE)
x192_2012	MORTALITY RATE, UNDER-5 (PER 1,000)
x243_2012	PROPORTION OF SEATS HELD BY WOMEN IN NATIONAL PARLIAMENTS (%)
x258_2012	RURAL POPULATION (% OF TOTAL POPULATION)
x283_2012	URBAN POPULATION (% OF TOTAL)
x58_2012 	BIRTH RATE, CRUDE (PER 1,000 PEOPLE)
'''

sub1 = data[['x142_2012','x126_2012','x156_2012','x167_2012','x192_2012',
             'x243_2012','x258_2012','x283_2012','x58_2012']]

sub2 = data[['x142_2013','x126_2013','x156_2013','x167_2013','x192_2013',
             'x243_2013','x258_2013','x283_2013','x58_2013']]

# Map codes to information for titles and captions
info = {'x142_2012':'GDP per capita',
        'x126_2012':'Broadband Subscriptions, per 100 people',
        'x156_2012':'Water Source, % of population with access',
        'x167_2012':'Internet Users, per 100 people',
        'x192_2012':'Under 5 Mortality Rate, per 1000 people',
        'x243_2012':'Parliament Seats Held by Women, %',
        'x258_2012':'Rural Population, % of population',
        'x283_2012':'Urban Population, % of population',
        'x58_2012':'Birth Rate, per 1000 people'}

# Set the variables to numeric
for variable in sub1:
    sub1[variable] = pd.to_numeric(sub1[variable], errors='coerce')
    

#==============================================================================
# Describe the variables
#==============================================================================
for variable in sub1:
    print (info[variable])
    print (sub1[variable].describe())
    print ('')


#==============================================================================
# Print univariate histograms
#==============================================================================
for variable in sub1:
    plt.figure()
    sns.distplot(sub1[variable].dropna(), kde=False);
    plt.ylabel('Count of countries')
    plt.title('Histogram for ' + info[variable])


#==============================================================================
# Bivariate scatterplots
#==============================================================================
for variable in sub1.columns[1:]:
    plt.figure()
    scat1 = sns.regplot(x="x142_2012", y=variable, fit_reg=False, data=sub1)
    plt.xlabel('GDP Per Capita')
    plt.ylabel(info[variable])
    plt.title('Scatterplot for the Association Between GDP Per Capita and ' 
              + info[variable])


#==============================================================================
# Bivariate scatterplots with correlation coefficient for regression analysis
#==============================================================================
import scipy.stats

# Create dataframe from original data with NaNs removed
data_clean = sub1.dropna()
# Create datafram from logarithms of original data
data_log = np.log(data_clean.astype('float64'))

# Inspect a linear relationship
for variable in sub1.columns[1:]:
    print ('Association between ' + info[variable] + ' and GDP Per Capita')
    print (scipy.stats.pearsonr(data_clean[variable], data_clean['x142_2012']))
    print ('')
    
    plt.figure()
    scat1 = sns.regplot(x="x142_2012", y=variable, fit_reg=True, data=data_clean)
    plt.xlabel('GDP Per Capita')
    plt.ylabel(info[variable])
    plt.title('Scatterplot for the Association Between GDP Per Capita and ' 
              + info[variable])


    # Inspect a logarithmic relationship
    print ('Association between log of ' + info[variable] + ' and GDP Per Capita')
    print (scipy.stats.pearsonr(data_log[variable], data_clean['x142_2012']))
    print ('')
    
    plt.figure()
    scat1 = sns.regplot(x="x142_2012", y=variable, fit_reg=True, data=data_log)
    plt.xlabel('GDP Per Capita')
    plt.ylabel('Log of ' + info[variable])
    plt.title('Scatterplot for the Association Between GDP Per Capita and log of ' 
              + info[variable])


#==============================================================================
# Run a Lasso Regression Analysis
#==============================================================================
from sklearn import preprocessing
from sklearn.linear_model import LassoLarsCV

# Select predictor variables and target variable as separate data sets  
predvar = data_clean[['x126_2012','x156_2012','x167_2012','x192_2012',
                      'x243_2012','x258_2012','x283_2012','x58_2012']]

target = data_clean.x142_2012
 
# Standardize predictors to have mean=0 and sd=1
predictors = predvar.copy()
for variable in predictors:
    predictors[variable]=preprocessing.scale(predictors[variable]
    .astype('float64'))


# Create test variables from corresponding 2013 data
data_clean2 = sub2.dropna()
target_test = data_clean2.x142_2013

predvar_test = data_clean2[['x126_2013','x156_2013','x167_2013','x192_2013',
                            'x243_2013','x258_2013','x283_2013','x58_2013']]

predictors_test = predvar_test.copy()
for variable in predictors_test:
    predictors_test[variable]=preprocessing.scale(predictors_test[variable]
    .astype('float64'))
    
pred_train = predictors
pred_test = predictors_test
tar_train = target
tar_test = target_test


# Specify the lasso regression model
model=LassoLarsCV(cv=10, precompute=False).fit(pred_train,tar_train)

# Print variable names and regression coefficients
print ('Regression coefficients')
for idx in range(0, len(predictors.columns)):
    print (predictors.columns[idx], ':', model.coef_[idx], 
           '    (' + info[predictors.columns[idx]] + ')')
#print (dict(zip(predictors.columns, model.coef_)))
print ('')

# Plot coefficient progression
plt.figure()
m_log_alphas = -np.log10(model.alphas_)
ax = plt.gca()
plt.plot(m_log_alphas, model.coef_path_.T)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha CV')
plt.ylabel('Regression Coefficients')
plt.xlabel('-log(alpha)')
plt.title('Regression Coefficients Progression for Lasso Paths')

# Plot mean square error for each fold
m_log_alphascv = -np.log10(model.cv_alphas_)
plt.figure()
plt.plot(m_log_alphascv, model.cv_mse_path_, ':')
plt.plot(m_log_alphascv, model.cv_mse_path_.mean(axis=-1), 'k',
         label='Average across the folds', linewidth=2)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha CV')
plt.legend()
plt.xlabel('-log(alpha)')
plt.ylabel('Mean squared error')
plt.title('Mean squared error on each fold')
         

# MSE from training and test data
from sklearn.metrics import mean_squared_error
train_error = mean_squared_error(tar_train, model.predict(pred_train))
test_error = mean_squared_error(tar_test, model.predict(pred_test))
print ('training data MSE')
print(train_error)
print ('')
print ('test data MSE')
print(test_error)
print ('')

# R-square from training and test data
rsquared_train=model.score(pred_train,tar_train)
rsquared_test=model.score(pred_test,tar_test)
print ('training data R-square')
print(rsquared_train)
print ('')
print ('test data R-square')
print(rsquared_test)
