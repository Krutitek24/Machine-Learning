""" This file is for Assignment#3 Task 1 "Test Your k-NN Implementation" Classification Assignment.This Python script is to show at least three runs of your implementation of k-NN on a single training
and testing split for three different 𝒌 values
The dataset I used is Song Popularity Dataset.This  data set contains information about the songs & it's popularity based on certain factors.


Kruti Patel(000857563),Machine Learning - COMP-10200, Mohawk College, Winter-2024"""

from sklearn.linear_model import LinearRegression, SGDRegressor
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate
from split_data import *
from sklearn.svm import SVR

## Output sizes of training and testing sets, and number of features

print(f"Training set size: {len(train_data)}, Testing set size: {len(test_data)}, Number of features: {train_data.shape[1]}")
print()

## LinearRegressor
print("LinearRegressor")
rgr = LinearRegression()#creating an instance of the LinearRegression class.
results = cross_validate(rgr, train_data, train_targets, scoring="neg_mean_squared_error",  cv=10)#Cross validation is used to evaluate machine learning models on the training set
print("Average MSE:",(-1*results["test_score"]).mean())# average mean squared error (MSE)
print("Minimum MSE:", (-1*results["test_score"]).min())# minimum MSE
print("Maximum MSE:", (-1*results["test_score"]).max())# maximum MSE
print()

## PolynomialRegressor
print("PolynomialRegressor")
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
degree=2 #polynomial degree
rgr=make_pipeline(PolynomialFeatures(degree),LinearRegression())# creating a pipeline
results = cross_validate(rgr, train_data, train_targets, scoring="neg_mean_squared_error",  cv=10)#using 10 folds to validate our model
print("Average MSE:",(-1*results["test_score"]).mean())#average mean squared error (MSE)
print("Minimum MSE:", (-1*results["test_score"]).min())# minimum MSE
print("Maximum MSE:", (-1*results["test_score"]).max())#maximum MSE
print()

## KneighborsRegressor
print("KneighborsRegressor")
rgr = KNeighborsRegressor(n_neighbors=28, weights='distance')# using k nearest neighbours algorithm with distance weighting.
results = cross_validate(rgr, train_data, train_targets, scoring="neg_mean_squared_error",  cv=10)#using 10 k fold cross validation
print("Average MSE:",(-1*results["test_score"]).mean()) #average mean squared error (MSE)
print("Minimum MSE:", (-1*results["test_score"]).min())# minimum MSE
print("Maximum MSE:", (-1*results["test_score"]).max())# maximum MSE
print()

##DecionTreeRegressor
print("DecionTreeRegressor")
rgr = DecisionTreeRegressor(max_depth=10, min_samples_leaf=6, min_samples_split=200)#creating an instance of the DecisionTreeRegeressor
results = cross_validate(rgr, train_data, train_targets, scoring="neg_mean_squared_error",  cv=10)#using 10 folds for validation
print("Average MSE:",(-1*results["test_score"]).mean())#average mean squared error (MSE)
print("Minimum MSE:", (-1*results["test_score"]).min())# minimum MSE
print("Maximum MSE:", (-1*results["test_score"]).max())# maximum MSE
print()

##Support VectorRegressor
print("Support VectorRegressor")
rgr = SVR(kernel='linear', C=50, gamma=0.001, epsilon=0.1)#creating an instance of the SVR class with linear kernel, regularization parameter C, gama
results = cross_validate(rgr, train_data, train_targets, scoring="neg_mean_squared_error",  cv=10)#using 10-fold cross validation
print("Average MSE:",(-1*results["test_score"]).mean())#average mean squared error (MSE)
print("Minimum MSE:", (-1*results["test_score"]).min())# minimum MSE
print("Maximum MSE:", (-1*results["test_score"]).max())# maximum MSE
print()

## RandomForestRegressor
print("RandomForestRegressor")
rgr = RandomForestRegressor(n_estimators=300,max_depth=10,max_features=12,random_state=5)#creating an instance of the RandomForest regressor
results = cross_validate(rgr, train_data, train_targets, scoring="neg_mean_squared_error",  cv=10)#using 10-fold cross validation
print("Average MSE:",(-1*results["test_score"]).mean())#average mean squared error (MSE)
print("Minimum MSE:", (-1*results["test_score"]).min())# minimum MSE
print("Maximum MSE:", (-1*results["test_score"]).max())# maximum MSE
print()
