
"""
This file is for Assignment#3 Regression Assignment.

The dataset I used is Song Popularity Dataset.This  data set contains information about the songs & it's popularity based on certain factors.
Here I use grid  search to find the best hyperparameters and then train a DecisionTreeRegressor model using those hyperparameters.
I will be splitting my data into training set and testing set, with 80% of the data going into the training set and the rest going
I also implement cross validation method to evaluate the performance of my trained DecisionTreeRegressor model.

Kruti Patel(000857563),Machine Learning - COMP-10200, Mohawk College, Winter-2024

"""
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from split_data import *

# The 'params' dictionary is defining the hyperparameters grid for the DecisionTreeRegressor model
# that will be used in the GridSearchCV. It specifies different values to be tested for the
# 'max_depth', 'min_samples_split', and 'min_samples_leaf' hyperparameters of the
# DecisionTreeRegressor model. The GridSearchCV will then search through all possible combinations of
# these hyperparameters to find the best set of hyperparameters that minimizes the mean squared error
# during cross-validation.
params = {
'max_depth': [None, 10, 20],
    'min_samples_split': [50,100,200],
    'min_samples_leaf': [2,4,6]

}

rgr = GridSearchCV(estimator=DecisionTreeRegressor(), param_grid=params, scoring="neg_mean_squared_error",cv=10)

## Train the Regressor
rgr = rgr.fit(train_data, train_targets)

## Results
print("Best Mean-Squared Error:", -1*rgr.best_score_)
print("Best Parameter Set:",rgr.best_params_)
print("best_model",rgr.best_estimator_)
