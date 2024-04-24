"""
This file is for Assignment#3 Regression Assignment.

The dataset I used is Song Popularity Dataset.This  data set contains information about the songs & it's popularity based on certain factors.
Here I use grid  search to find the best hyperparameters and then train a Random Forest Regressor model using those hyperparameters.
I will be splitting my data into training set and testing set, with 80% of the data going into the training set and the rest going
I also implement cross validation method to evaluate the performance of my trained Random Forest Regressor model.

Kruti Patel(000857563),Machine Learning - COMP-10200, Mohawk College, Winter-2024

"""
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from split_data import *
# The `params` dictionary is defining the hyperparameters grid for the Random Forest Regressor model.
# It specifies different values to be tested for the 'max_features', 'n_estimators', and 'max_depth'
# hyperparameters during the grid search process. The 'GridSearchCV' will search through all possible
# combinations of these hyperparameters to find the best set of values that minimizes the mean squared
# error on the training data.
params = {
            "max_features": [4, 8, 12],
            "n_estimators": [150,200,300,400,500],
            "max_depth": [5, 10, 15, 20, 25]
}
rgr = GridSearchCV(estimator=RandomForestRegressor(), param_grid=params, scoring="neg_mean_squared_error",cv=10)

## Train the Regressor
rgr = rgr.fit(train_data, train_targets)
## Results
print("Best Mean-Squared Error:", -1*rgr.best_score_)
print("Best Parameter Set:",rgr.best_params_)
print("best_model",rgr.best_estimator_)




