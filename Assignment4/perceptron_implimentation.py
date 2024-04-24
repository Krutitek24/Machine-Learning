"""
This file is for Assignment#4 Neural Networks Assignment.
The code provided in this assignment is a simple implementation of perceptron alorithem.

#### Data in Datasets 000857563_2.csv, 000857563_3.csv are linarly seperatable.
#### Data in Datasets 000857563_1.csv, 000857563_4.csv are not linarly seperatable.

Kruti Patel(000857563),Machine Learning - COMP-10200, Mohawk College, Winter-2024

"""
import numpy as np

from sklearn.model_selection import train_test_split

# Function to make predictions using the perceptron model
def predict(ws, t, xs):
    '''The function `predict` takes in weights, a threshold, and inputs, and returns whether the dot
    product of weights and inputs is greater than the threshold.
    
    Parameters
    ----------
    ws
        The parameter `ws` represents the weights for each input feature in a machine learning model.
    t
        The parameter `t` in the `predict` function represents the threshold value that the dot product of
    weights (`ws`) and inputs (`xs`) needs to exceed in order for the function to return `True`.
    xs
        The parameter `xs` in the `predict` function represents the inputs or features that you want to
    make predictions on. These could be a list or array of numerical values that correspond to the
    features of your data.
    
    Returns
    -------
        The function `predict` is returning a boolean value, which is the result of comparing the dot
    product of weights (`ws`) and inputs (`xs`) with the threshold (`t`).
    
    '''
   
    return np.dot(ws, xs) > t

# Function to train the perceptron
def train_perceptron(X_train, y_train, num_features, epochs,lr=0.1):
    '''The function `train_perceptron` trains a perceptron model using the perceptron learning algorithm
    with a defined number of epochs and learning rate.
    
    Parameters
    ----------
    X_train
        X_train is the training data containing input features. 
    y_train
        The `y_train` parameter in the `train_perceptron` function represents the target labels or classes
    for the corresponding examples in the training dataset `X_train`. These target labels are used to
    train the perceptron model by adjusting the weights and threshold based on the predicted output
    compared to the actual target
    num_features
        The `num_features` parameter in the `train_perceptron` function represents the number of features
    in the input data. It is used to initialize the weights vector with zeros of the appropriate size
    for the perceptron algorithm. The weights vector will have a length equal to the number of features
    in the
    epochs
        The `epochs` parameter in the `train_perceptron` function refers to the number of times the
    algorithm will loop over the entire training dataset during the training process. 
    lr
        Learning rate (lr): It is a hyperparameter that controls the step size during the training of the
    perceptron. It determines how much the weights and threshold are updated in each iteration of the
    training process. A higher learning rate can lead to faster convergence but may also result in
    overshooting the optimal solution
    
    Returns
    -------
        The function `train_perceptron` returns the updated weights (`ws`) and threshold (`t`) after
    training the perceptron model on the provided training data (`X_train`, `y_train`) for the specified
    number of epochs.
    
    '''    
    ws = np.zeros(num_features)# Initialize weights to zeros and threshold to zero
    t = 0

    # Loop over the dataset for a defined number of epochs
    for epoch in range(epochs):
        for xs, target in zip(X_train, y_train):
            # Predict the output for the current example
            output = predict(ws, t, xs)
            # Update rules for perceptron learning algorithm
            if output < target:
                ws += xs * lr  # Update weights
                t -= lr  # Update threshold
            elif output > target:
                ws -= xs * lr  # Update weights
                t += lr  # Update threshold
    return ws, t

# Function to load dataset, train and test the perceptron, and report performance
def load_and_train_test(file):
    '''This function is designed to load a file and train a model for testing purposes.
    
    Parameters
    ----------
    file
        The "file" parameter in the function "load_and_train_test" likely refers to the file path or name
    of the file that contains the data to be loaded and used for training and testing a machine learning
    model. This file could be a CSV file, a text file, or any other format that
    
    '''
    # Load the data from the file into a numpy array
    dataset = np.genfromtxt(file, delimiter=',')
    # Separate the features  and target
    data = dataset[:, :-1]
    target = dataset[:, -1].astype(int)
    # Split training and testing data.
    train, test, train_target, test_target =train_test_split(data, target, test_size = 0.2)

    # Train the perceptron model on the training set
    ws, t = train_perceptron(train, train_target, train.shape[1],1000)

    # Test the perceptron model and make predictions on the test set
    predictions = np.array([predict(ws, t, xs) for xs in test])
    # Calculate accuracy as the percentage of correct predictions
    accuracy = np.mean(predictions == test_target) * 100

    # Return the accuracy, final weights, and threshold
    return accuracy, ws, t

# List of datasets to process
dataset_names = ["000857563_1.csv", "000857563_2.csv", "000857563_3.csv", "000857563_4.csv"]

results = []

# Loop through each dataset, train and test the perceptron, and print results
for dataset_name in dataset_names:
    accuracy, ws, t = load_and_train_test(dataset_name)
    results.append(f'Name" "{dataset_name}\nAccurcy: {accuracy:.2f}% \nWeights:{ws} \nThrashold:{t:.1f}\n')
# Print the results for each dataset
for result in results:
    print(result)


