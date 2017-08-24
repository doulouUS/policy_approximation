import numpy as np
from sklearn import svm
import math
import scipy
import time


import read_data_fedex as rdf

# training param
percentage = 0.7

# Get the sets of entries and labels for training and Test
whole_data = rdf.data_prep_2nd_attempt()

# input
indices = np.asarray(whole_data[1])
rows = indices[:, 0]
columns = indices[:, 1]
input_data = np.asarray(scipy.sparse.coo_matrix((whole_data[0], (rows, columns))).todense())

# shuffling
nb_samples = input_data.shape[0]
arr = np.arange(nb_samples)
np.random.shuffle(arr)

input_data = input_data[arr, :]
# labels
label_data = np.asarray(whole_data[3])
label_data = label_data[arr, 1]
# print("labelsLLLLLLL ", label_data)

print("-------")
print("Dataset summary ")
print("shape of entries ", input_data.shape)
print("shape of labels", label_data.shape)
print('#######')

# Train and test data
train_entries = input_data[:math.floor(percentage*input_data.shape[0]), :]
train_labels = label_data[:math.floor(percentage*input_data.shape[0])]
print("Training set ")
print("Shape of training data : ", train_entries.shape)
print("Shape of labels        :", train_labels.shape)

test_entries = input_data[math.floor(percentage*input_data.shape[0]):, :]
test_labels = label_data[math.floor(percentage*input_data.shape[0]):]
print("Shape of test data : ", test_entries.shape)
print("Shape of test labels        :", test_labels.shape)

# data_set_train = rdf.Dataset(train_entries, train_labels)
# data_set_test = rdf.Dataset(test_entries, test_labels)

# SVM model
start_time = time.time()
svc = svm.NuSVC(
    nu=0.01,
    kernel='rbf',
    decision_function_shape=None,
    cache_size=1000,
    tol=8e-5,
    probability=False
).fit(train_entries, train_labels)
end_time = time.time()
print("Training time: ", end_time - start_time)
# print("Test with first test entry  : ", data_set_train.entries[0])
# print("label is  : ", data_set_train.labels[0])

pred = svc.predict(test_entries)

# pred_prob = svc.predict_proba(test_entries)
# idx_sorted = np.argsort(pred_prob)


# Accuracy
errors = pred - test_labels

"""
# Accuracy on 3 best predicted classes
count = 0
for i in range(0, idx_sorted.shape[0]):
    print(idx_sorted[i, 0:3])
    print(test_labels[i])
    if test_labels[i] in idx_sorted[i, 0:3]:
        count+=1
"""
print("Accuracy on test set: ", np.count_nonzero(errors==0)/test_labels.shape[0])


# Best score obtained after moving various parameter: 29% with rbf, tol=8e-5 and nu = 0.01

