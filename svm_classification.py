import numpy as np
from sklearn import svm
import math


import read_data_fedex as rdf

# Get the sets of entries and labels for training and Test
whole_data = rdf.ReadDataFedex()
percentage_training = 0.7
num_examples = whole_data.num_examples

# Train and test data
train_entries = whole_data.entries[:math.floor(percentage_training*num_examples)]
train_labels = whole_data.labels[:math.floor(percentage_training*num_examples)]

test_entries = whole_data.entries[math.floor(percentage_training*num_examples):]
test_labels = whole_data.labels[math.floor(percentage_training * num_examples):]

data_set_train = rdf.Dataset(train_entries, train_labels)
data_set_test = rdf.Dataset(test_entries, test_labels)

# SVM model
C = 1.
svc = svm.SVC(kernel='linear', C=C).fit(data_set_train.entries, data_set_train.labels)
pred = svc.predict(data_set_test.entries)
print("Accuracy on test set: ", pred.shape[0] - np.count_nonzero(pred - data_set_test.labels) )