"""
This file is for Assignment#3 Regression Assignment.In this assignment, I Get the song_data.csv Data File and then I  extract features and targets.
The target is the first column of the csv file, while all other columns represent features. Here I normalise the data as well
The dataset I used is Song Popularity Dataset.This  data set contains information about the songs & it's popularity based on certain factors.
Here

Kruti Patel(000857563),Machine Learning - COMP-10200, Mohawk College, Winter-2024
"""
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import MinMaxScaler
## Load the Song Popularity Dataset and shuffle it
raw = np.genfromtxt("song_data.csv", delimiter="," ,skip_header=1)
raw = shuffle(raw)
data = raw[:,1:]
targets = raw[:,0]# Target variable is in the first column of dataset
## Normalize data using Min Max Scalar
scaler = MinMaxScaler()
data = scaler.fit_transform(data)
## Train/Test Split
train_data, test_data, train_targets, test_targets = train_test_split(data, targets,test_size = 0.2)


