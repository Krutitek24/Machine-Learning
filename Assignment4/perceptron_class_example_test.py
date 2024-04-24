"""
This file is for Assignment#4 Neural Networks Assignment.
This the code for testing  own implimentation with  Class example.

Kruti Patel(000857563),Machine Learning - COMP-10200, Mohawk College, Winter-2024

"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def predict(ws, t, xs):
    return np.dot(ws, xs) > t

def train_perceptron(X_train, y_train, num_features,  epochs,lr=0.1):
    '''This function trains a perceptron model using the given training data and parameters.

    Parameters
    ----------
    X_train
        X_train is the input training data, where each row represents a training example and each column
    represents a feature.
    y_train
        The `train_perceptron` function seems to be a function for training a perceptron model. The
    parameters are as follows:
    num_features
        The `num_features` parameter represents the number of features in your input data. It is used to
    determine the size of the weight vector in the perceptron algorithm.
    epochs
        The parameter "epochs" in the function "train_perceptron" refers to the number of times the
    algorithm will iterate over the entire training dataset during the training process. Each iteration
    over the entire dataset is called an epoch. Increasing the number of epochs allows the model to
    learn more from the training data
    lr
        Learning rate for updating the weights during training.

    '''

    ws = np.zeros(num_features)
    t = 0

    for epoch in range(epochs):#for each epoch in the number of iterations defined
        for xs, target in zip(X_train, y_train):#go through every single sample in the dataset
            output = predict(ws, t, xs)#get the predicted value based on current w and b
            if output < target:
                ws += xs * lr # update weights
                t -= lr #update threshold
            elif output > target:
                ws -= xs * lr # update weights
                t += lr #update threshold.
    return ws, t

def load_and_train_test(file):
    '''The function loads a dataset, trains a perceptron model on the data, and returns the accuracy,
    weights, and threshold of the model.

    Parameters
    ----------
    dataset_name
        The function `load_and_train_test` seems to be loading a dataset, training a perceptron model on
    the data, and then testing the model's accuracy. However, it seems like there are some missing parts
    in the code such as the implementation of the `train_perceptron` and `predict

    Returns
    -------
        The function `load_and_train_test` returns the accuracy of the perceptron model on the dataset, as
    well as the weights (ws) and threshold (t) learned during training.

    '''
    # Load the data from the file into a numpy array
    dataset = np.genfromtxt(file, delimiter=',')
    # Separate the features  and target
    X = dataset[:, :-1]
    Y = dataset[:, -1].astype(int)
    ws, t = train_perceptron(X, Y, X.shape[1],100)

    # Testing
    predictions = np.array([predict(ws, t, xs) for xs in X])# Make predictions using the trained model
    accuracy = np.mean(predictions == Y) * 100# Calculate the accuracy of the model

    return accuracy, ws, t


file="test.csv"
accuracy, ws, t = load_and_train_test("test.csv")#
result=(f'Name" "{file}\n Accurcy: {accuracy:.2f}% \nWeights:{ws} \nThrashold:{t:.1f}\n')
print(result)
