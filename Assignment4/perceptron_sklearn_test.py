"""
This file is for Assignment#4 Neural Networks Assignment.
This the code for compairing the  performance own implimentation with  SKLearn Perceptron.

Kruti Patel(000857563),Machine Learning - COMP-10200, Mohawk College, Winter-2024

"""
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import numpy as np

from sklearn.model_selection import train_test_split
# Function to load data, train the Perceptron, and calculate accuracy
def train_test_perceptron_sklearn(file):
    # Load the data from the file into a numpy array
    dataset = np.genfromtxt(file, delimiter=',')
    # Separate the features  and target
    data = dataset[:, :-1]
    target = dataset[:, -1].astype(int)
    # Split training and testing data.
    train, test, train_target, test_target = train_test_split(data, target, test_size = 0.2)

    # Initialize the Perceptron model
    perceptron = Perceptron( max_iter=1000,eta0=0.01)

    # Train the Perceptron model
    perceptron.fit(train, train_target)

    # Make predictions on the testing set
    predictions = perceptron.predict(test)

    # Calculate and print the accuracy
    accuracy = accuracy_score(test_target, predictions)
    print(f'{dataset_name}: Accuracy = {accuracy*100:.2f}%')

# Example usage with your datasets
dataset_names = ["000857563_1.csv", "000857563_2.csv", "000857563_3.csv", "000857563_4.csv"]

for dataset_name in dataset_names:
    train_test_perceptron_sklearn(dataset_name)
