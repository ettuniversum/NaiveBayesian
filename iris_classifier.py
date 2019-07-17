# Iris

# Want to build my own Nieve Bayes Classifier
# Use Normal Distrabution (Gaussian)
# Use Iris Data Set through Sci Kit Learn

# Python version: 3.5.4
# Numpy version: 1.14.0
# SciPy version: 1.0.0
# SciKit-Learn version : 0.19.1

import numpy as np
import scipy as sp
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# creating the iris dataset opbject
iris_dataset = load_iris()

# inpsect iris_dataset keys
print ()
print ("Iris Dataset Keys:")
print (iris_dataset.keys())

# X is original data
# rows are individual samples
# colums are features
X = iris_dataset.data

# class label (integer)
# corresponds to index of row of X
y = iris_dataset.target

# inspect target names
print ()
print ("Class Labels are:")
print (iris_dataset.target_names)

# number of class labels
Cn = len(iris_dataset.target_names)
print ("{} class labels".format(Cn))

# n number of samples
# d is number of features
n,d_features = X.shape
print ()
print ("We have {} samples".format(n))
print ("With {} features each".format(d_features))

# Random partition into a training set and a test set
# default is to assign %75 percent of samples to train and %25 to test
X_train, X_test, y_train, y_test = train_test_split(X, y)
n_train = len(X_train)
n_test = len(X_test)
print ()
print ("{} samples in the training set".format(n_train))
print ("{} samples in the test set".format(n_test))

# mu_mat will contain the mean of the data
# each row represents the class
# colums corespond to features
mu_mat = np.zeros((Cn,d_features), dtype=float)

# sig_mat will contain the standard deviation of the data by class
# each row represents the class
# colum corespond to features
sig_mat = np.zeros((Cn,d_features), dtype=float)

# use the training set to create our model
# calculate mu_mat and sig_mat
# For each class label we will look at a subset of our training set
# to calculate the mean and standard deviation
for i in range(0,Cn):
	mu_mat[i] = X_train[y_train==i].mean(axis=0)
	sig_mat[i] = X_train[y_train==i].std(axis=0)

# The areah under the normal/gaussian curve represents the probability
# we will use a small sliver(2*epsilon) to aproximate the probability.
epsilon = 0.01

# class_likelihood (Prior)
class_likelihood = np.zeros(Cn, dtype=float)
# Loop through class labels
for i in range(0,Cn):
    # Sum distribution each column of data for probability
	class_likelihood[i] = (y_train==i).sum()/n_train

# Class_given_data (Posterior)
# Intialize class results
class_given_data_mat = np.zeros((n_test,Cn), dtype=float)
# Loop through row of data
for i in range(0,n_test):
    # Loop through class labels
	for k in range(0,Cn):
		class_given_data = np.log(class_likelihood[k])
        # Loop through the feature colums
		for j in range(0,d_features):
			class_given_data = class_given_data + np.log(sp.stats.norm.pdf(X_test[i,j],mu_mat[k,j],sig_mat[k,j])*2*epsilon)
		class_given_data_mat[i,k] = class_given_data

print ()
print ("Probability that the Model predicts the correct class label:")
print ((class_given_data_mat.argmax(axis=1)==y_test).sum()/n_test)
print ()
