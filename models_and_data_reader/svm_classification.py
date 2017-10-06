import math
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy
from sklearn import svm, tree

from models_and_data_reader import read_data_fedex as rdf

def train_svm_model(nb_truck, method="classification", percentage=0.7):
    # TODO store models to avoid recomputing it each time
    # TODO Finish the regression by implementing the policy argmax(...)
    """
    Return a scikit learn model (SVM), to take a decision based on a state.
    :param method: str, "classification" or "regression"
    :param percentage: float, split train-test set
    :return: svm model
    """
    # Get the sets of entries and labels for training and Test
    print("*_*_*_*_*_*_* Data preparation *_*_*_*_*_*_*_*_*")
    whole_data = rdf.data_prep_2nd_attempt(nb_truck=nb_truck,
                                           method=method)
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
    if method == "classification":
        label_data = np.asarray(whole_data[3])
        label_data = label_data[arr, 1]
        # print("labelsLLLLLLL ", label_data)

    elif method == "regression":
        label_data = np.asarray(whole_data[2])
        label_data = np.asarray(label_data[arr])

    print("-------")
    print("Dataset summary ")
    print("shape of entries ", input_data.shape)
    print("shape of labels", label_data.shape)
    print("*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_**_*_")

    # Train and test data
    train_entries = input_data[:math.floor(percentage*input_data.shape[0]), :]
    train_labels = label_data[:math.floor(percentage*input_data.shape[0])]
    print("Training set ")
    print("Shape of training data : ", train_entries.shape)
    print("Shape of labels        : ", train_labels.shape)

    if percentage < 1:
        test_entries = input_data[math.floor(percentage*input_data.shape[0]):, :]
        test_labels = label_data[math.floor(percentage*input_data.shape[0]):]
        print("*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_**_*_")
        print("Training set ")
        print("Shape of test data : ", test_entries.shape)
        print("Shape of test labels        :", test_labels.shape)

    # data_set_train = rdf.Dataset(train_entries, train_labels)
    # data_set_test = rdf.Dataset(test_entries, test_labels)

    if method == "classification":
        # SVM model
        start_time = time.time()
        print("*_*_*_*_*_*_* TRAINING *_*_*_*_*_*_*")
        model = svm.NuSVC(
            nu=0.01,
            kernel='rbf',
            decision_function_shape=None,
            cache_size=1000,
            tol=8e-5,
            probability=True
        ).fit(train_entries, train_labels)
        end_time = time.time()
        print("Training time: ", end_time - start_time)
        # print("Test with first test entry  : ", data_set_train.entries[0])
        # print("label is  : ", data_set_train.labels[0])

        if percentage > 1:
            pred = model.predict(test_entries)

            # Accuracy
            errors = pred - test_labels

            """
            # Accuracy on 3 best predicted classes
            pred_prob = model.predict_proba(test_entries)
            idx_sorted = np.argsort(pred_prob)

            count = 0
            for i in range(0, idx_sorted.shape[0]):
                print(idx_sorted[i, 0:3])
                print(test_labels[i])
                if test_labels[i] in idx_sorted[i, 0:3]:
                    count+=1
            """
            """
            # Accuracy on best predicted class among remaining jobs
            pred_prob = model.predict_proba(test_entries)
            idx_sorted = np.argsort(pred_prob)

            count = 0
            for i in range(0, idx_sorted.shape[0]):
                print(idx_sorted[i, 0:3])
                print(test_labels[i])
                if test_labels[i] in idx_sorted[i, 0:3]:
                    count+=1
            """
            print("Accuracy on test set: ", np.count_nonzero(errors == 0)/test_labels.shape[0])

            # Best score obtained after moving various parameter: 29% with rbf, tol=8e-5 and nu = 0.01

    elif method == "regression":

        # standardization
        '''
        scale_entries = preprocessing.StandardScaler()
        train_entries = scale_entries.fit_transform(train_entries)
        test_entries = scale_entries.fit_transform(test_entries)

        scale_labels = preprocessing.StandardScaler()
        train_labels = scale_labels.fit_transform(train_labels)
        test_labels = scale_labels.fit_transform(test_labels)
        '''
        start_time = time.time()
        # model = svm.NuSVR(C=0.5).fit(train_entries, train_labels)
        # model = linear_model.ElasticNet(l1_ratio=0.5,  fit_intercept=True,normalize=False ).fit(train_entries, train_labels)

        # Interesting one: test under 40% of error: 82% of the samples! (with or without standardization)
        # model = linear_model.ElasticNetCV(l1_ratio=[.01, .05, .1, .5, .7, .9, .95, .99, 1],  fit_intercept=True,normalize=False ).fit(train_entries, train_labels)

        # Even better, almost too good to be true... TODO understand it!
        model = tree.DecisionTreeRegressor().fit(train_entries, train_labels)

        # NN with potentially several layers

        # clustering => Dirichlet process mixtures

        # features selection? http://scikit-learn.org/stable/modules/feature_selection.html

        end_time = time.time()
        print("Training time: ", end_time - start_time)

        # print("check test input ", test_entries[0,:])
        print("check label train", test_entries[0])
        print("check label test ", test_labels[0])

        pred_train = model.predict(train_entries)
        pred_test = model.predict(test_entries)

        print("Prediction train", pred_train[0])
        print("Prediction test", pred_test[0])

        # Accuracy
        errors_test = np.divide(np.asarray(pred_test - test_labels), test_labels)
        errors_train = np.divide(np.asarray(pred_train - train_labels), train_labels)

        print("Accuracy on train set: ", np.count_nonzero(errors_train==0)/train_labels.shape[0])
        print("Number of test under 10 of absolute error", np.where(np.logical_and(errors_train >= -.10, errors_train <= .10))[0].shape[0]/train_labels.shape[0])
        print("Number of test under 20 of absolute error", np.where(np.logical_and(errors_train >= -.20, errors_train <= .20))[0].shape[0]/train_labels.shape[0])
        print("Number of test under 30 of absolute error", np.where(np.logical_and(errors_train >= -.30, errors_train <= .30))[0].shape[0]/train_labels.shape[0])
        print("Number of test under 40 of absolute error", np.where(np.logical_and(errors_train >= -.40, errors_train <= .40))[0].shape[0]/train_labels.shape[0])

        print("Accuracy on test set: ", np.count_nonzero(errors_test==0)/test_labels.shape[0])
        print("Number of test under 10 of absolute error", np.where(np.logical_and(errors_test >= -.10, errors_test <= .10))[0].shape[0]/test_labels.shape[0])
        print("Number of test under 20 of absolute error", np.where(np.logical_and(errors_test >= -.20, errors_test <= .20))[0].shape[0]/test_labels.shape[0])
        print("Number of test under 30 of absolute error", np.where(np.logical_and(errors_test >= -.30, errors_test <= .30))[0].shape[0]/test_labels.shape[0])
        print("Number of test under 40 of absolute error", np.where(np.logical_and(errors_test >= -.40, errors_test <= .40))[0].shape[0]/test_labels.shape[0])

        plt.bar(range(len(errors_test)), errors_test, color="blue")

        plt.show()

    return model

if __name__ == "__main__":
    train_svm_model()
