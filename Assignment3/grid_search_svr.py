"""
This file is for Assignment#3 Regression Assignment.
The dataset I used is Song Popularity Dataset.This  data set contains information about the songs & it's popularity based on certain factors.
Here I use grid  search to find the best hyperparameters and then train a model SVR using those hyperparameters.
I will be splitting my data into training set and testing set, with 80% of the data going into the training set and the rest going
I also implement cross validation method to evaluate the performance of my trained model for Support Vector Regressor.
Kruti Patel(000857563),Machine Learning - COMP-10200, Mohawk College, Winter-2024
"""
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from split_data import *
# The 'params' dictionary in your code is defining the hyperparameters grid for the Support Vector
# Regression (SVR) model that will be used in the GridSearchCV. Each key in the dictionary corresponds
# to a hyperparameter of the SVR model, and the associated list contains the values that GridSearchCV
# will iterate over to find the best combination of hyperparameters.
params = {

        'kernel': [ 'linear','rbf', 'poly'],
        'C': [0.1, 1, 50, 100],
        'gamma': [ 0.01,0.1, 1],
        'epsilon': [0.1, 0.2, 0.5, 1.0]
}
rgr = GridSearchCV(estimator=SVR(), param_grid=params, scoring="neg_mean_squared_error",cv=10)

## Train the Regressor
rgr = rgr.fit(train_data, train_targets)

## Results
print("Best Mean-Squared Error:", -1*rgr.best_score_)
print("Best Parameter Set:",rgr.best_params_)
print("best_model",rgr.best_estimator_)