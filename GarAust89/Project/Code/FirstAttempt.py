# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 13:32:43 2015

@author: garauste
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.preprocessing import Imputer
from statsmodels.graphics.mosaicplot import mosaic
from sklearn import ensemble
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import  GridSearchCV
from sklearn.feature_selection import RFECV
from sklearn import tree
import matplotlib.pyplot as plt



## Read in the training dataset
df = pd.read_csv("C:\\Users\\garauste\\Dropbox\\General Assembly\\Project\\Titanic\\Titanic Data\\train.csv")
df.head()

'''
VARIABLE DESCRIPTIONS:
survival        Survival
                (0 = No; 1 = Yes)
pclass          Passenger Class
                (1 = 1st; 2 = 2nd; 3 = 3rd)
name            Name
sex             Sex
age             Age
sibsp           Number of Siblings/Spouses Aboard
parch           Number of Parents/Children Aboard
ticket          Ticket Number
fare            Passenger Fare
cabin           Cabin
embarked        Port of Embarkation
                (C = Cherbourg; Q = Queenstown; S = Southampton)
'''

########
## Performing some EDA of the data
########

# Pull out survived as the response series
response_series = df.Survived

# Create some Mosiac plots to inspect the data
mosaic(df,['Pclass','Survived'], title = 'Survival Rate by Class')
mosaic(df,['Sex','Survived'], title = 'Survival Rate by Gender')

## Creating a function to pull out the titles of the Passengers

def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first ) + 1
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""

# Test the function to pull out the titles
test_title = find_between(df.Name[100],',','.')
print test_title
# The function seems to be working correctly. Now we need to apply it to the 
# entire dataframe.
titles = list()
Names = list(set(df.Name))
titles = [find_between(e, ',', '.') for e in df.Name] 

# Check out a list of unique titles: 
unique_titles = list(set(titles))
people_per_title = [(k, sum(1 for e in titles if e == k)) for k in unique_titles]
# The majority of titles fall into Mr., Miss., Mrs., and Master. 

# Bucket some of the titles as Highborns #
Highborn = ['Sir','Major','the Countess','Don','Jonkheer','Col','Lady','Dr','Rev','Capt']

# Assigning the HighBorn Status
for e in titles:
    if e in Highborn:
        titles[titles.index(e)] = 'Highborn'

# Check out the titles again and the number of people per title
unique_titles = list(set(titles))
people_per_title = [(k, sum(1 for e in titles if e == k)) for k in unique_titles]

# Next let's turn the one Mme and MS title into Mrs. 
other_womens = ['Ms','Mme']
# Reassign these to women
for e in titles:
    if e in other_womens:
        titles[titles.index(e)] = 'Mrs'
        
# Assigning the Mlle title of Madamemoiselle to Miss
mlle = ['Mlle']
for e in titles:
    if e in mlle:
        titles[titles.index(e)] = 'Miss'

# Assign the new titles variable to df
df['Titles'] = titles

# Let's get a mosaic of survival rates by title
mosaic(df,['Titles','Survived'], title = 'Survival Rate by Title')

# Determine the ratio of men to women on boards
men_onboard = df.Sex[df.Sex == 'male'].count().astype(np.float)
# 577 men on board
women_onboard = df.Sex[df.Sex == 'female'].count().astype(np.float)
# 314 women on board

# Determine women survivors to male survivors
male_survivors = df.Survived[df.Sex=='male'].astype(np.float).sum()
# 109 Men survived 
female_survivors = df.Survived[df.Sex=='female'].astype(np.float).sum()
# 233 women survived

# Get the survival percentages
percent_female_survivors = female_survivors/women_onboard
percent_male_survivors = male_survivors/men_onboard
# 74% of women on board survived compared to only 18.8% of men

# Get counts of the embarkation ports
df.Embarked.count()
unique_ports = list(set(df.Embarked))

# Get counts
embarkers = [(k,sum(1 for e in df.Embarked if e == k)) for k in unique_ports]
print embarkers
# Most people embarked at Southampton, create dummies for the other two ports, 2 
# nan embarkers will be assigned to Southampton
df.Embarked['nan'] = 'S'

df['Q_Port'] = 0 
df.Q_Port[df.Embarked=='Q'] = 1

df['C_Port'] = 0 
df.C_Port[df.Embarked=='C'] = 1

# Get a list of unique ages
uni_ages = list(set(df.Age))
        

Ages = np.array(df.Age)

# Create an imputer object
imputer_object = Imputer(missing_values='NaN', strategy='mean', axis=1)
imputer_object.fit(Ages)
df['Ages'] = pd.DataFrame(imputer_object.transform(Ages).reshape(891,1)) 

# Female Indicator
df['Female'] = 0
df.Female[df.Sex == 'female'] = 1

# Inspect the data again.
df.head()

## Let's create a basic linear regression
est = smf.ols(formula = 'Survived ~ Pclass + Ages + SibSp + Parch + Q_Port + C_Port + Female', data = df).fit()
# Print out the summary
print est.summary()
# R-Squared is currently very poor - all variables are significant with most coefficient as expected
# Parch and Q_Port are the only unsignificant variables

#########
# Further Feature Engineering
#########

# Create a feature variable for ages and class
df['AgeByClass'] = df.Ages*df.Pclass

# Create a family size variable
df['Family_Size'] = df.Parch + df.SibSp

# Rerun the linear regression
est_v2 = smf.ols(formula = '''Survived ~ Pclass + Ages + SibSp + Parch + Q_Port 
                            + C_Port + Female + Family_Size + AgeByClass''', data = df).fit()

# Print the summary of the new model
print est_v2.summary()
# This is still terrible - time to test a logistic regression

# Logistic regression model
logist = smf.logit(formula = '''Survived ~ Female + C_Port + Q_Port''', data = df).fit()

# Print out logistic regression results
print logist.summary()

#####
# Creating binary variables for logistic regression
#####

##
# Create a histogram of Ages in order to create buckets
## 
df['Ages'].hist(bins=16, alpha = 0.5)

# Create to dummies - elderly for over 60s and children for under 15s
df['Elderly'] = 0
df.Elderly[df.Ages >55] = 1

df['Children'] = 0
df.Children[df.Ages < 15] = 1

# Histogram of classes
df['Pclass'].hist(bins=3, alpha = 0.5)
# Create dummies for 1st or second class

df['First_Class'] = 0
df.First_Class[df.Pclass == 1] = 1

df['Second_Class'] = 0
df.Second_Class[df.Pclass == 2] = 1

# Use binary features variable from class to bin the titles
def get_binary_values(data_frame):
    """encodes the categorical features in Pandas
    """
    all_columns = pd.DataFrame(index=data_frame.index)
    for col in data_frame.columns:
        data = pd.get_dummies(data_frame[col], prefix=col.encode('ascii','replace'))
        all_columns = pd.concat([all_columns,data],axis=1)
    return all_columns

# Bin Titles
df['Mrs'] = 0
df.Mrs[df.Titles == 'Mrs'] = 1

df['Miss'] = 0
df.Miss[df.Titles == 'Miss'] = 1

df['Highborn'] = 0
df.Highborn[df.Titles == 'Highborn'] = 1

df['Master'] = 0
df.Master[df.Titles == 'Master'] = 1


###
# Histogram of Family Size
###
df['Family_Size'].hist(bins = 16, alpha = 0.5) 
# Create a dummy for family size greater than 5

df['large_family'] = 0
df.large_family[df.Family_Size >5] = 1

##
# Run Logistic Regression Again
##

logist_v2 = smf.logit(formula = '''Survived ~ Female + C_Port + Q_Port + Elderly
                                   + Children + First_Class + Second_Class + 
                                   large_family''', data = df).fit()
                                   
# print out summary
print logist_v2.summary()                                     
# Drop all categorical varaibles before proceeding
df = df.drop(['Name','Sex','Age','Ticket','PassengerId','Cabin','Fare','Embarked','Titles','Pclass','Survived'], axis = 1)



############
## PCA
############

from sklearn.decomposition import PCA

pca = PCA(n_components = 6)
pca.fit(df)

# extract the components
pca_df = pd.DataFrame(pca.transform(df))

## plotting the first two principal components
pca_df.plot(x=0, y=1, kind = 'scatter')


# Making a scree plot
variance_df = pd.DataFrame({'variance': pca.explained_variance_, 'principal component':
    pca_df.columns.tolist()})
    
# adding one to pricnipal componetns (since there is no 0th compeonet)
variance_df['principal component'] = variance_df['principal component'] + 1
variance_df.plot(x='principal component', y = 'variance')
#  looks like variance stops getting explained after first two components 

pca_df_small = pca_df.ix[:,0:1]

# getting a cross val score of transformed data
rf = ensemble.RandomForestClassifier(n_estimators = 500)
roc_scores_rf_pca = cross_val_score(rf, pca_df_small, response_series, cv = 10,
                                    scoring = 'roc_auc')
                                    
print roc_scores_rf_pca.mean() 
# 73% accuracy 

# compare to the original data
roc_scores_rf = cross_val_score(rf, df, response_series, cv = 10,
                                scoring = 'roc_auc') 
print roc_scores_rf.mean() 
# 95% accuracy - PCA created information loss

############################
# Support Vector Machines 
############################

from sklearn.svm import SVC

# first running the quadratic kernel with PCA

svm = SVC(kernel = 'poly')  

roc_scores_svm = cross_val_score(svm, df, response_series, cv=10,
                                 scoring='roc_auc', verbose = 1)                              
print roc_scores_svm.mean()                                 

# let's do a grid search
param_grid = dict(kernel = ['linear','poly','rbf','sigmoid'])    
 
svm_grid = GridSearchCV(svm, param_grid, cv=10, scoring='roc_auc')
svm_grid.fit(df,response_series)
best_estimator = svm_grid.best_estimator_
print best_estimator.kernel
 
# rbf won
print svm_grid.best_score_                                   
#==============================================================================
#==============================================================================
# best estimator was 94% - just below RFs
# Note: SVMs are more accurate than RFs with trending data! 


##############
## Random Forests
##############

# Creating a random forest object
## these are the default values of the classifier
rf = ensemble.RandomForestClassifier(n_estimators = 10,
criterion='gini',max_depth=None,min_samples_split = 2, min_samples_leaf=1,max_features='auto',
max_leaf_nodes=None,bootstrap = True, oob_score=False, random_state=None,verbose=0,min_density=None,
compute_importances=None)

# I'm ging to change this a bit
rf = ensemble.RandomForestClassifier(n_estimators=500)

## let's compute ROC AUC of the randome forest
roc_scores_rf = cross_val_score(rf,df, response_series, cv=10, 
                                scoring = 'roc_auc')
                            
# DO the same for a decision tree
roc_score_tree = cross_val_score(tree.DecisionTreeClassifier(),df,
 response_series, cv=10, scoring='roc_auc')
 
# Compare the mean
print roc_scores_rf.mean()
print roc_score_tree.mean()
# RF massive outperformance

# create new class with a .coef_ attribute
class RFClassifierWithCoef(ensemble.RandomForestClassifier):
   def fit(self, *args, **kwargs):
       super(ensemble.RandomForestClassifier, self).fit(*args, **kwargs)
       self.coef_ = self.feature_importances_

rf_with_coef = RFClassifierWithCoef(n_estimators = 40,criterion='gini',max_depth=None,
min_samples_split = 2, min_samples_leaf=1,max_features='auto',max_leaf_nodes=None,
bootstrap = True, oob_score=False, random_state=None,verbose=0,min_density=None,
compute_importances=None)

rfe_cv = RFECV(estimator = rf_with_coef, step = 1, cv = 10, scoring = 'roc_auc',verbose = 1)
rfe_cv.fit(df, response_series)

print "Optimal number of features: {0} of {1} considered".format(rfe_cv.n_features_,
len(df.columns))

# pritning out socres as we increas the number of features -- the farther down the list
# the higher the number of features considered.
print rfe_cv.grid_scores_

# let's plot out the results
plt.figure()
plt.xlabel('Number of Features selected')
plt.ylabel('Cross Validation score (ROC_AUC)')
plt.plot(range(1, len(rfe_cv.grid_scores_)+1),rfe_cv.grid_scores_)
plt.show()

features_used = df.columns[rfe_cv.get_support()]
print features_used

# you can extract the final selected model object his way
final_estimator_used = rfe_cv.estimator_

# perform grid search to find the optimal number of trees

trees_range = range(10,750,10)
param_grid = dict(estimator__n_estimators = trees_range)


grid_rf = GridSearchCV(rfe_cv, param_grid, cv=10, scoring = 'roc_auc', verbose = 1)
grid_rf.fit(df, response_series)
# check out the scores of the grid search
grid_rf_mean_scores = [result[1] for result in grid_rf.grid_rf_scores_]

# plot the results of the grid search
plt.figure()
plt.plot(trees_range, grid_rf_mean_scores)
plt.hold(True)
plt.plot(grid_rf.best_params_['estimator__max_depth'],
         grid_rf.best_score_,'ro',markersize=12, markeredgewidth=1.5,
         markerfacecolor='None',markeredgecolor='r')
plt.grid(True)

# pull out the best estimator and print it's ROC AUC
best_rf_est = grid_rf.best_estimator_
# how many trees did the best estimator have
print best_rf_est.n_estimators
# how accurate was the best estimator
print grid_rf.best_score_


#########################
## Boosting Trees 
#########################
boosting_tree = ensemble.GradientBoostingClassifier()

roc_scores_gbm = cross_val_score(boosting_tree, df, response_series,
                                 cv = 10, scoring = 'roc_auc')
                                 
# Compare the accuracies
print roc_scores_gbm.mean()
print roc_scores_rf.mean()
print roc_score_tree.mean()


class GBMClassifierWithCoef(ensemble.GradientBoostingClassifier):
   def fit(self, *args, **kwargs):
       super(ensemble.RandomForestClassifier, self).fit(*args, **kwargs)
       self.coef_ = self.feature_importances_


gbm_with_coef = GBMClassifierWithCoef()



rfe_cv = RFECV(estimator = gbm_with_coef, step = 1, cv = 10, scoring = 'roc_auc',verbose = 1)
rfe_cv.fit(df, response_series)

print "Optimal number of features: {0} of {1} considered".format(rfe_cv.n_features_,
len(df.columns))

# let's plot out the results
plt.figure()
plt.xlabel('Number of Features selected')
plt.ylabel('Cross Validation score (ROC_AUC)')
plt.plot(range(1, len(rfe_cv.grid_scores_)+1),rfe_cv.grid_scores_)
plt.show()


# Let's tune for num_trees, learning_rate, and subsampling percent.
# need to import arrange to create ranges for floats
from numpy import arange

learning_rate_range = arange (0.01, 0.6, 0.02)
subsampling_range = arange(0.25, 1, 0.2)
n_estimators_range = arange(10,100,10)

param_grid = dict(learning_rate = learning_rate_range, n_estimators=
n_estimators_range, subsample = subsampling_range)

gbm_grid = GridSearchCV(boosting_tree, param_grid, cv=10, scoring = 'roc_auc', verbose = 1)
gbm_grid.fit(df, response_series)

best_gbm = gbm_grid.best_params_

# find the winning paramters
print gbm_grid.best_params_
# how dos this compare to the default settings
# estimators = 100, sumsample = 1.00, learning rate = 0.1

# pull out best score
print gbm_grid.best_score_
print grid.best_score_

# ROC Curve of GBM vs RF vs Tree method
from sklearn.cross_validation import train_test_split
from sklearn import metrics

xTrain, xTest, yTrain, yTest = train_test_split(df, response_series, test_size=0.3)

tree_probabilities = pd.DataFrame(tree.DecisionTreeClassifier().fit(xTrain,yTrain).predict_proba(xTest))
rf_probabilities = pd.DataFrame(best_rf_est.fit(xTrain,yTrain).predict_proba(xTest))
gbm_probabilities = pd.DataFrame(gbm_grid.best_estimator_.fit(xTrain,yTrain).predict_proba(xTest))

tree_fpr, tree_tpr, thresholds = metrics.roc_curve(yTest, tree_probabilities[1])
rf_fpr, rf_tpr, thresholds = metrics.roc_curve(yTest, rf_probabilities[1])
gbm_fpr, gbm_tpr, thresholds = metrics.roc_curve(yTest, gbm_probabilities[1])

plt.figure()
dt, = plt.plot(tree_fpr, tree_tpr, color = 'g')
rf, = plt.plot(rf_fpr, rf_tpr, color = 'b')
gbm, = plt.plot(gbm_fpr, gbm_tpr, color = 'r')
plt.xlabel('False Positive Rates (1-Specificity)')
plt.ylabel('True Positives Rate (Sesitivity)')
plt.legend([dt, rf, gbm], ['Decision Tree','Random Forest','Boosting Tree'])
# From this plot we see that we can 100% correctly classify but will incorrectly 
# classify 10% of others

# Create partial dependence plot on most important features for gbm

importances = pd.DataFrame(gbm_grid.best_estimator_.feature_importances_, 
index = df.columns, columns = ['importance'])

importances.sort(columns=['importance'], ascending=False, inplace = True)
print importances

from sklearn.ensemble.partial_dependence import plot_partial_dependence

features = [i for i,j in enumerate(df.columns.tolist()) if j in
importances.importance[0:3].index.tolist()]

fix, axs = plot_partial_dependence(gbm_grid.best_estimator_, df,
features, feature_names = df.columns)


####################
## Neural Networks
####################

from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn import linear_model

logistic_classifier = linear_model.LogisticRegression()
neural_net = BernoulliRBM(random_state=0,verbose=True)

# create a pipeline of a neural net connected to a logistic regression
neural_classifier = Pipeline(steps=[('neural_net',neural_net),('logistic_classifier',
logistic_classifier)])

# you can cross-validate the entire pipeline like any old classifier
roc_scores_nn = cross_val_score(neural_classifier, df, response_series,
cv=10, scoring='roc_auc')

# lET'S compare accracies
print roc_scores_nn.mean()
print roc_scores_gbm.mean()
print roc_scores_rf.mean()
print roc_score_tree.mean()

## Ranges
learning_rate_range = arange(0.01,0.2,0.05)
iteration_range = range(30,50,5)
components_range = range(250,500,50)


# Grid Search time
param_grid = dict(neural_net__n_components = components_range,
neural_net__learning_rate = learning_rate_range
, neural_net__n_iter = iteration_range
)

# do 5 fold grid search
nn_grid = GridSearchCV(neural_classifier, param_grid, cv=5, scoring = 'roc_auc')
nn_grid.fit(df,response_series)

# Pull out best score
print nn_grid.best_score_
# compare to other grid best scores
print gbm_grid.best_score_
print grid.best_score_


print nn_grid.best_params_


################################
# Read in the testing set and prep it
################################

## Read in the training dataset
df_test = pd.read_csv("C:\\Users\\garauste\\Dropbox\\General Assembly\\Project\\Titanic\\Titanic Data\\test.csv")
df_test.head()

df_submit = df_test

# Test the function to pull out the titles
test_title = find_between(df_test.Name[100],',','.')
print test_title
# The function seems to be working correctly. Now we need to apply it to the 
# entire dataframe.
titles = list()
Names = list(set(df_test.Name))
titles = [find_between(e, ',', '.') for e in df_test.Name] 

# Check out a list of unique titles: 
unique_titles = list(set(titles))
people_per_title = [(k, sum(1 for e in titles if e == k)) for k in unique_titles]
# The majority of titles fall into Mr., Miss., Mrs., and Master. 

# Bucket some of the titles as Highborns #
Highborn = ['Sir','Major','the Countess','Don','Jonkheer','Col','Lady','Dr','Rev','Capt']

# Create a new title column and leave it blank 
title_new = []
# Assing the HighBorn Status
for e in titles:
    if e in Highborn:
        titles[titles.index(e)] = 'Highborn'

# Check out the titles again and the number of people per title
unique_titles = list(set(titles))
people_per_title = [(k, sum(1 for e in titles if e == k)) for k in unique_titles]

# Next let's turn the one Mme and MS title into Mrs. 
other_womens = ['Ms','Mme']
# Reassign these to women
for e in titles:
    if e in other_womens:
        titles[titles.index(e)] = 'Mrs'
        
# Assigning the Mlle title of Madamemoiselle to Miss
mlle = ['Mlle']
for e in titles:
    if e in mlle:
        titles[titles.index(e)] = 'Miss'

# Assign the new titles variable to df
df_test['Titles'] = titles


# Determine the ratio of men to women on boards
men_onboard = df_test.Sex[df_test.Sex == 'male'].count().astype(np.float)
# 577 men on board
women_onboard = df_test.Sex[df_test.Sex == 'female'].count().astype(np.float)
# 314 women on board

# Get the survival percentage
# Get counts of the embarkation ports
df_test.Embarked.count()
unique_ports = list(set(df_test.Embarked))

# Get counts
embarkers = [(k,sum(1 for e in df_test.Embarked if e == k)) for k in unique_ports]
print embarkers
# Most people embarked at Southampton, create dummies for the other two ports, 2 
# nan embarkers will be assigned to Southampton
df_test.Embarked['nan'] = 'S'

df_test['Q_Port'] = 0 
df_test.Q_Port[df_test.Embarked=='Q'] = 1

df_test['C_Port'] = 0 
df_test.C_Port[df_test.Embarked=='C'] = 1

# Get a list of unique ages
uni_ages = list(set(df_test.Age))


Ages = np.array(df_test.Age)

# Create an imputer object
imputer_object = Imputer(missing_values='NaN', strategy='mean', axis=1)
imputer_object.fit(Ages)
df_test['Ages'] = pd.DataFrame(imputer_object.transform(Ages).reshape(418,1)) 

# Female Indicator
df_test['Female'] = 0
df_test.Female[df_test.Sex == 'female'] = 1

#########
# Further Feature Engineering
#########

# Create a feature variable for ages and class
df_test['AgeByClass'] = df_test.Ages*df_test.Pclass

# Create a family size variable
df_test['Family_Size'] = df_test.Parch + df_test.SibSp

#####
# Creating binary variables for logistic regression
#####

# Create to dummies - elderly for over 60s and children for under 15s
df_test['Elderly'] = 0
df_test.Elderly[df_test.Ages >55] = 1

df_test['Children'] = 0
df_test.Children[df_test.Ages < 15] = 1

df_test['First_Class'] = 0
df_test.First_Class[df_test.Pclass == 1] = 1

df_test['Second_Class'] = 0
df_test.Second_Class[df_test.Pclass == 2] = 1

# Bin Titles
df_test['Mrs'] = 0
df_test.Mrs[df_test.Titles == 'Mrs'] = 1

df_test['Miss'] = 0
df_test.Miss[df_test.Titles == 'Miss'] = 1

df_test['Highborn'] = 0
df_test.Highborn[df_test.Titles == 'Highborn'] = 1

df_test['Master'] = 0
df_test.Master[df_test.Titles == 'Master'] = 1

df_test['large_family'] = 0
df_test.large_family[df_test.Family_Size >5] = 1

# Drop all categorical varaibles before proceeding
df_test = df_test.drop(['Name','Sex','Age','Ticket','PassengerId','Cabin','Fare','Embarked','Titles','Pclass'], axis = 1)


# Get predictions
# gbm_best 

test_survivors_gbm = list(gbm_grid.predict(df_test))
df_submit['Survived'] = test_survivors_gbm

# Drop all unnecssary columns
df_submit = df_submit.drop(['Name','Sex','Age','Ticket','Cabin',
                            'Fare','Embarked','Titles','Pclass','Mrs','Highborn',
                            'Master','Elderly','Children','First_Class','Second_Class',
                            'Miss','large_family','SibSp','Parch','Ages','Q_Port',
                            'C_Port','AgeByClass','Family_Size'], axis = 1)

# Write results to csv
df_submit.to_csv('C:\\Users\\garauste\\Dropbox\\General Assembly\\Project\\Titanic\\Results\\GBM_Preds.csv')