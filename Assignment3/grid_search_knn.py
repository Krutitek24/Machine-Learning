"""
This file is for Assignment#3 Regression Assignment.

The dataset I used is Song Popularity Dataset.This  data set contains information about the songs & it's popularity based on certain factors.
Here I use grid  search to find the best hyperparameters and then train a K-Nearest Neighbors Regressor model using those hyperparameters.
I will be splitting my data into training set and testing set, with 80% of the data going into the training set and the rest going
I also implement cross validation method to evaluate the performance of my trained K-Nearest Neighbors Regressor model.

Kruti Patel(000857563),Machine Learning - COMP-10200, Mohawk College, Winter-2024

"""
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from split_data import *
# This code block is defining a dictionary named 'params' that contains different hyperparameters for
# the K-Nearest Neighbors Regressor model. The hyperparameters specified in the dictionary are:
params = {
    "n_neighbors": [23,25,28],
    "weights": ["uniform", "distance"],
    "p": [1, 2]
}
rgr = GridSearchCV(estimator=KNeighborsRegressor(), param_grid=params, scoring="neg_mean_squared_error",cv=10)

## Train the Regressor
rgr = rgr.fit(train_data, train_targets)

## Results
print("Best Mean-Squared Error:", -1*rgr.best_score_)
print("Best Parameter Set:",rgr.best_params_)
print("best_model",rgr.best_estimator_)

