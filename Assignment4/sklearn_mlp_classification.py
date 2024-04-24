"""
This file is for Assignment#4 Neural Networks Assignment.

This script is designed to process and analyze five datasets: four artificial datasets and one selected from
either the UCI Machine Learning Repository. The main objective is to apply a Multi-Layer Perceptron (MLP)
model to each dataset using  SKLearn .

The tasks performed by this script are as follows:
- Read each dataset from its respective file.
- Initialize and train an MLP model on each dataset.
  For classification tasks, the performance metric is accuracy.
  - Record and output the following details for each dataset:
  * Accuracy
  * The number of iterations until the model converges.
  * The configuration of the MLP network, including:
    - The number of nodes in each hidden layer.
    - Any other non-default settings used in the model.

Requirements:
- SKLearn  must be installed and available for importing.
- Data files for the four artificial datasets and the one chosen dataset must be accessible.

Note:
- The configuration and optimization of the MLP model might vary depending on the dataset's characteristics

Kruti Patel(000857563),Machine Learning - COMP-10200, Mohawk College, Winter-2024

"""

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from sklearn.tree import DecisionTreeClassifier

dataset_names = ["000857563_1.csv","000857563_2.csv","000857563_3.csv","000857563_4.csv"]
file_names=["000857563_1.csv","000857563_2.csv","000857563_3.csv","000857563_4.csv"]
results = [{'max_iter': 1000, 'activation': 'relu', 'alpha': 0.001, 'learning_rate': 'adaptive', 'solver': 'adam'},{'max_iter': 500},{'max_iter': 500},{'max_iter': 1000, 'activation': 'relu', 'alpha': 0.001, 'learning_rate': 'adaptive', 'solver': 'sgd'}]


# This for loop is iterating over three lists simultaneously: `dataset_names`, `results`, and
# `file_names`.
for dataset_name,result,file_name in zip(dataset_names,results,file_names):

    # Load the data from the file into a numpy array
    dataset_name = np.genfromtxt(dataset_name, delimiter=',')
    # Separate the features  and target
    data = dataset_name[:, :-1]
    target = dataset_name[:, -1].astype(int)
    # Normalization code
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    # Split training and testing data.
    train_data, test_data, train_target, test_target =train_test_split(data, target, test_size = 0.2)
    #results.append(params_set)
    mlp = MLPClassifier(**result) # add some parameters here
    mlp.fit(train_data, train_target)
    print("-----------------------")
    print("File: ",file_name)
    print("-----------------------")
    ## Test the Classifier
    y_pred = mlp.predict(test_data)
    a = accuracy_score(test_target, y_pred)
    print(" MLP: ",result)
    print("Number of Epochs Used:",mlp.n_iter_)

    print()
    print(f"    Accuracy (test set): {a*100:.2f}%")
    y_pred = mlp.predict(train_data)
    a = accuracy_score(train_target, y_pred)
    print(f"    Accuracy (train set): {a*100:.2f}%")

    # Initialize the Decision Tree Classifier
    clf = DecisionTreeClassifier()

    # Train the model
    clf.fit(train_data, train_target)
    # Make predictions on the test set
    predictions = clf.predict(test_data)

    # Evaluate the model
    accuracy = accuracy_score(test_target, predictions)
    print(f"    DecisionTree Accuracy: {accuracy*100:.2f}%")
    print()

## Load a data set
dataset_name1 = np.genfromtxt("train_data.csv",skip_header=1, delimiter=',')
# Separate the features  and target
train_data = dataset_name1[:, :-1]
train_target = dataset_name1[:, -1].astype(int)
# Normalization code
scaler = StandardScaler()
data=scaler.fit_transform(train_data)

dataset_name2 = np.genfromtxt("test_data.csv",skip_header=1, delimiter=',')
# Separate the features  and target
test_data = dataset_name2[:, :-1]
test_target = dataset_name2[:, -1].astype(int)
# Normalization code
scaler = StandardScaler()
data=scaler.fit_transform(test_data)
## Create and train a Multi-Layer Perceptron
params_set = {  'max_iter' :10000,
            'activation': 'relu',
            'alpha': 0.001,
            'learning_rate': 'adaptive',
            'solver': 'sgd',
            'hidden_layer_sizes':(6,7)
        }
mlp = MLPClassifier(**params_set) # add some parameters here
mlp.fit(train_data, train_target)
params = mlp.get_params()
print()
print("--------------------------------------")
print("File: train_data.csv   test_data.csv")
print("--------------------------------------")

## Test the Classifier
y_pred = mlp.predict(test_data)
a = accuracy_score(test_target, y_pred)
print("Number of Epochs Used:",mlp.n_iter_)
print(" MLP: ",params_set)
print()
print(f"    Accuracy (test set): {a*100:.2f}%")
y_pred = mlp.predict(train_data)
a = accuracy_score(train_target, y_pred)
print(f"    Accuracy (train set): {a*100:.2f}%")

# Initialize the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42,criterion='entropy',max_depth=20,min_samples_leaf=4)

# Train the model
clf.fit(train_data, train_target)
# Make predictions on the test set
predictions = clf.predict(test_data)

# Evaluate the model
accuracy = accuracy_score(test_target, predictions)
print(f"    DecisionTree Accuracy: {accuracy*100:.2f}%")


