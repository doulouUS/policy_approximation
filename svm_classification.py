import numpy as np
from sklearn import svm
import math
import scipy
import time


import read_data_fedex as rdf

# TODO The kernel trick!
# TODO RBF too!

# Get the sets of entries and labels for training and Test
whole_data = rdf.ReadDataFedex()
percentage_training = 0.01
num_examples = whole_data.num_examples

# Train and test data
train_entries = scipy.sparse.csr_matrix(whole_data.entries[:math.floor(percentage_training*num_examples)], dtype=np.float64)
train_labels = scipy.sparse.csr_matrix(whole_data.labels[:math.floor(percentage_training*num_examples)], dtype=np.float64)

print("Shape of training data : ", train_entries.shape)

test_entries = scipy.sparse.csr_matrix(whole_data.entries[
               math.floor(percentage_training*num_examples):math.floor(percentage_training*num_examples)+100
               ], dtype=np.float64)
test_labels = scipy.sparse.csr_matrix(whole_data.labels[
              math.floor(percentage_training * num_examples):math.floor(percentage_training*num_examples)+100
              ], dtype=np.float64)

# data_set_train = rdf.Dataset(train_entries, train_labels)
# data_set_test = rdf.Dataset(test_entries, test_labels)

# SVM model
C = 1.
start_time = time.time()
svc = svm.NuSVC(nu=0.01, cache_size=1000).fit(train_entries, train_labels)
end_time = time.time()
print("Training time: ", end_time - start_time)
# print("Test with first test entry  : ", data_set_train.entries[0])
# print("label is  : ", data_set_train.labels[0])
pred = svc.predict(test_entries)
# print("Predicted  : ", pred)
errors = pred - test_labels
print("Accuracy on test set: ", np.count_nonzero(errors==0))
print("With total number of entries ", 100)


# (pred.shape[0] - np.count_nonzero(errors==0))/pred.shape[0])