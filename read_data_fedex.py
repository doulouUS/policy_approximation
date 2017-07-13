import pandas as pd
import numpy as np
import pickle
import itertools
import time
import math
from scipy.sparse.coo import coo_matrix

PATH_TO_DATA = "/Users/Louis/PycharmProjects/MEng_Research/foo-Environment_2/dynamics/demand_models/fedex.data"
PATH_TO_ADDRESSES = "/Users/Louis/PycharmProjects/MEng_Research/foo-Environment_2/gym_foo/envs/addresses.fedex"


class Dataset:
    """
    Will allow to access, prepare fedex data in order to be fed into a neural net
    """

    def __init__(self, path):

        self.path = path
        self.dataframe = pd.read_csv(
            path,
            header=0,
            delim_whitespace=True
        )

        # interesting values
        self.nb_entries = len(self.dataframe["PostalCode"])
        self.nb_addresses = int(self.dataframe.loc[self.dataframe['Address'].idxmax()]["Address"])+1
        self.nb_postal_code = len(set(list(self.dataframe["PostalCode"])))

        self.end_time = int(self.dataframe.loc[self.dataframe['StopStartTime'].idxmax()]["StopStartTime"])
        self.start_time = int(self.dataframe.loc[self.dataframe['StopStartTime'].idxmin()]["StopStartTime"])

        # interesting columns
        self.stop_order_col = self.dataframe["StopOrder"]
        self.cur_time_col = self.dataframe["StopStartTime"]
        self.addresses_id_col = self.dataframe["Address"]
        self.postal_code_col = self.dataframe["PostalCode"]
        self.pickup_type_col = self.dataframe["PickupType"]

        self.start_tour_idx = list(self.stop_order_col[self.stop_order_col == 1 ].index)

        # All the data (take up to 3 mn to load)

        indices, values, dense_shape, indices_label, values_label = self.get_sparse_rpz_loc(
            [1, len(self.start_tour_idx) - 1],
            column="address",  # "postal_code" is not supported yet
            cur_loc_code=1,
            rem_deliv_code=0.5,
            rem_pickup_code=-0.25
        )

        indices = np.asarray(indices)
        rows = indices[:, 0]
        columns = indices[:, 1]
        sparse_entries = coo_matrix((values, (rows, columns)), shape=dense_shape)
        # input data
        self.entries = np.asarray(sparse_entries.todense(), dtype=np.float16)

        indices_l = np.asarray(indices_label)
        # rows_l = indices_l[:, 0]
        columns_l = indices_l[:, 1]
        # sparse_labels = coo_matrix((values_label, (rows_l, columns_l)), shape=dense_shape)

        # label data => of dimension [num_examples], not one-hot encoded !
        # uint16->0-65000
        self.labels = np.asarray(columns_l, dtype=np.uint16)  # sparse_labels.todense(), dtype=np.int8)
        print("Shape of the written labels ", self.labels.shape)
        self.num_examples = dense_shape[0]


        # Follow training with these params
        self.epochs_completed = 0
        self.index_in_epoch = 0

    def address_id_to_string(self, id):
        """

        :param id: int
        :return:
        """
        with open(PATH_TO_ADDRESSES, "rb") as f:
            addresses = pickle.load(f)

        return addresses[id]

    def get_batch(self, range_tour):
        """
        Return training example
        :return:
        """
        NotImplemented

    def get_rem_deliv(self, range):
        """
        Return deliveries within a range of indexes "range", from self.dataframe

        :param range: list of 2 int
        :return: list
        """
        sub_data = self.dataframe[range[0]:range[1]]
        return sub_data[sub_data["PickupType"] == 0]

    def get_rem_pickup(self, range):
        """
        Return pickups within a range of indexes "range", from self.dataframe

        :param range: list of 2 int
        :return: list
        """
        sub_data = self.dataframe[range[0]:range[1]]
        return sub_data[sub_data["PickupType"] != 0]

    def get_sparse_rpz_loc(self, range_tour, column, cur_loc_code=1, rem_deliv_code=0.25, rem_pickup_code=-0.25):
        """
        Return the 3 components to build a sparse matrix described as follows:
        matrix of type [nb of training examples, nb of features]

        as well as to build a sparse matrix of labels

        [ self.nb_entries * self.nb_addresses ] if features="addresses" AND cur_loc = False
        [ self.nb_entries * self.nb_postal_code ] if features="postal_code" AND cur_loc = False

        [ self.nb_entries * (self.nb_addresses + 1)] if features="addresses" AND cur_loc = True
        [ self.nb_entries * (self.nb_postal_code + 1)] if features="postal_code" AND cur_loc = True

        :param range_tour: list, number of operation tours retrieved (1 tour = 1 truck operating)
            column: str, retrieve locations by Addresses or Postal Codes
            *_code: float, score assigned in the generalized one-hot encoding

        :return: indices, values, dense_shape
        """
        beginning = time.time()
        # parameters to create a sparse tensor
        indices = []
        values = []

        indices_label = []
        values_label = []

        array_line = 0

        for start, end in itertools.zip_longest(
                self.start_tour_idx[range_tour[0]-1:range_tour[1]],
                self.start_tour_idx[range_tour[0]:range_tour[1]+1],
                fillvalue=0
        ):
            # start: beginning of truck's trip
            # end: ...

            # print('start ', start)
            # print('end', end)

            if end != 0:
                for step in range(end - start - 1):
                    # cur loc
                    cur_loc = self.addresses_id_col[start + step]
                    # print("loc ",cur_loc)
                    # ...
                    # print("CUR LOC ", cur_loc)

                    if column == "address":
                        # remaining deliveries
                        rem_deliv = list(self.get_rem_deliv([start + step + 1, end])["Address"])
                        # print("rem_deliv ", rem_deliv)

                        # remaining pickups => maybe use POSTAL CODE instead
                        rem_pick = list(self.get_rem_pickup([start + step + 1, end])["Address"])
                        # print("rem_pick ", rem_pick)

                        # add successively delivery, pickup, current location
                        # adapted to the SparseTensor requirements
                        indices.extend([[array_line, i] for i in rem_deliv])
                        indices.extend([[array_line, i] for i in rem_pick])
                        indices.extend([[array_line, cur_loc]])

                        values.extend([rem_deliv_code] * len(rem_deliv))
                        values.extend([rem_pickup_code] * len(rem_pick))
                        values.extend([cur_loc_code])

                    elif column == "postal_code":
                        # TODO ENCODE postal codes !! mapping from each postal code to an int (\in [1, ~16 000]
                        # remaining deliveries
                        rem_deliv = list(self.get_rem_deliv([start + step + 1, end])["PostalCode"])
                        # print("rem_deliv ", rem_deliv)

                        # remaining pickups => maybe use POSTAL CODE instead
                        rem_pick = list(self.get_rem_pickup([start + step + 1, end])["PostalCode"])

                        # add successively delivery, pickup, current location
                        # adapted to the SparseTensor requirements
                        indices.extend([[array_line, i] for i in rem_deliv])
                        indices.extend([[array_line, i] for i in rem_pick])
                        indices.extend([[array_line, cur_loc]])

                        values.extend([rem_deliv_code] * len(rem_deliv))
                        values.extend([rem_pickup_code] * len(rem_pick))
                        values.extend([cur_loc_code])

                    # label
                    next_loc = self.addresses_id_col[start + step + 1]

                    indices_label.extend([[array_line, next_loc]])

                    values_label.extend([1])
                    array_line += 1
                    # print("LABEL ", next_loc)

        if column == "address":
            dense_shape = [indices[-1][0]+1, self.nb_addresses]

        elif column == "postal_code":
            dense_shape = [indices[-1][0]+1, self.nb_postal_code]

        ending = time.time()
        print("___________________________________________________")
        print("Data initialized. Summary:")
        print("  ")
        print("Number of training examples:      ", dense_shape[0])
        print("Number of features:               ", dense_shape[1])
        if column == "address":
            print("Locations referred to as:           Addresses")
        elif column == "postal_code":
            print("Locations referred to as:           Postal Codes")

        print("  ")
        print("Time elapsed during preparation:  ", ending - beginning)
        print("  ")
        print("___________________________________________________")

        return indices, values, dense_shape, indices_label, values_label


class ReadDataFedex:
    "Fast access to pre-saved data + help for manipulation (generate batches etc.)"

    def __init__(self):

        print("____________________________________________________________________________")
        print(" ")
        print(" DATA LOADING...")
        loaded = np.load('dataset.fedex.npz')
        self.entries = loaded['inputs']
        self.labels = loaded['labels']

        print(" DATA SUCCESSFULLY LOADED ! ")
        print("Shape of the entries:    ", self.entries.shape)
        print("Shape of the labels:    ", self.labels.shape)
        print(" ")

        self.num_examples = self.entries.shape[0]
        self.num_features = self.entries.shape[1]

        print("____________________________________________________________________________")


class TrainAndTest(ReadDataFedex):

    def __init__(self, percentage=0.7):
        """

        :param percentage: percentage, float. Proportion of dataset used for training
        """

        # to follow training
        self.index_in_epoch_train = 0
        self.epochs_completed_train = 0

        self.index_in_epoch_test = 0
        self.epochs_completed_test = 0

        self.percentage = percentage

        # Root dataset
        self.root_data = ReadDataFedex()

        # TRAINING DATASET
        length = self.root_data.entries.shape[0]
        self.entries_train = self.root_data.entries[:math.floor(percentage*length), :]
        self.labels_train = self.root_data.labels[:math.floor(percentage*length)]

        self.num_examples_train = self.entries_train.shape[0]
        # self.num_features_train = self.entries_train.shape[1]

        # TESTING DATASET
        self.entries_test = self.root_data.entries[math.floor(percentage*length):, :]
        self.labels_test = self.root_data.labels[math.floor(percentage*length):]

        self.num_examples_test = self.entries_test.shape[0]

    def next_batch(self, batch_size, task="train_eval", shuffle=True):
        # TODO Cette imple est degeulasse avec ce mot cles "task" qui double le code, a reprendre a la sauce Google
        # TODO il n'y a pas que ca. pleins de choses a revoir...
        """Return the next `batch_size` examples from our TRAINING data set contained in self.entries_sparse_param.

        :param task, str. say if the function has to give a batch of training "train_eval" or testing "test_eval"
        """
        if task == "train_eval":
            start_ = self.index_in_epoch_train

            # Shuffle for the first epoch

            if self.epochs_completed_train == 0 and start_ == 0 and shuffle:
                perm0 = np.arange(self.num_examples_train)
                np.random.shuffle(perm0)
                # TODO self.entries ici?
                self.entries = self.entries_train[perm0, :]
                self.labels = self.labels_train[perm0]

            # Go to the next epoch
            if start_ + batch_size > self.num_examples_train:
                # Finished epoch
                self.epochs_completed_train += 1
                # Get the rest examples in this epoch
                rest_num_examples = self.num_examples_train - start_
                entries_rest_part = self.entries_train[start_:self.num_examples_train, :]
                labels_rest_part = self.labels_train[start_:self.num_examples_train]
                # Shuffle the data
                if shuffle:
                    perm = np.arange(self.num_examples_train)
                    np.random.shuffle(perm)
                    self.entries_train = self.entries_train[perm, :]
                    self.labels_train = self.labels_train[perm]
                # Start next epoch
                start_ = 0
                self.index_in_epoch_train = batch_size - rest_num_examples
                end = self.index_in_epoch_train
                entries_new_part = self.entries_train[start_:end, :]
                labels_new_part = self.labels_train[start_:end]
                result = np.concatenate((entries_rest_part, entries_new_part), axis=0), np.concatenate(
                    (labels_rest_part, labels_new_part), axis=0)

                return result

            else:
                self.index_in_epoch_train += batch_size
                end = self.index_in_epoch_train
                return self.entries_train[start_:end, :], self.labels_train[start_:end]

        elif task == "test_eval":
            start_ = self.index_in_epoch_test

            # Shuffle for the first epoch

            if self.epochs_completed_test == 0 and start_ == 0 and shuffle:
                perm0 = np.arange(self.num_examples_test)
                np.random.shuffle(perm0)
                self.entries = self.entries_test[perm0, :]
                self.labels = self.labels_test[perm0]

            # Go to the next epoch
            if start_ + batch_size > self.num_examples_test:
                # Finished epoch
                self.epochs_completed_test += 1
                # Get the rest examples in this epoch
                rest_num_examples = self.num_examples_test - start_
                entries_rest_part = self.entries_test[start_:self.num_examples_test, :]
                labels_rest_part = self.labels_test[start_:self.num_examples_test]
                # Shuffle the data
                if shuffle:
                    perm = np.arange(self.num_examples_test)
                    np.random.shuffle(perm)
                    self.entries_test = self.entries_test[perm, :]
                    self.labels_test = self.labels_test[perm]
                # Start next epoch
                start_ = 0
                self.index_in_epoch_test = batch_size - rest_num_examples
                end = self.index_in_epoch_test
                entries_new_part = self.entries_test[start_:end, :]
                labels_new_part = self.labels_test[start_:end]
                result = np.concatenate((entries_rest_part, entries_new_part), axis=0), np.concatenate(
                    (labels_rest_part, labels_new_part), axis=0)

                return result

            else:
                self.index_in_epoch_test += batch_size
                end = self.index_in_epoch_test
                return self.entries_test[start_:end, :], self.labels_test[start_:end]



if __name__ == "__main__":

    # this is looong
    fedex = Dataset(PATH_TO_DATA)
    np.savez_compressed("dataset.fedex", inputs=fedex.entries, labels=fedex.labels)
    """
    start = time.time()
    data = TrainAndTest(0.1)
    print("all data shape |", data.root_data.num_examples)
    print("train shape    |", data.num_examples_train)
    end = time.time()

    print("Elapsed time ", end - start)

    # Small size test

    print("list of start ", fedex.start_tour_idx)
    print("address with 0 as ID? ", fedex.dataframe[fedex.dataframe["Address"] == 0])
    start = time.time()
    ids, vls, ds_s, ids_l, vls_l = fedex.get_sparse_rpz_loc([1,3-1], "address")
    end = time.time()

    print("Time elapsed: ", end - start)

    indices = np.asarray(ids)
    rows = indices[:, 0]
    columns = indices[:, 1]

    indices_l = np.asarray(ids_l)
    rows_l = indices_l[:, 0]
    columns_l = indices_l[:, 1]
    print("rows shape ", rows)
    print("columns shape ", columns)
    print("values shape", len(vls))
    print("shape ", ds_s)
    sparse_entries = coo_matrix((vls_l, (rows_l, columns_l)), shape=ds_s)
    print(sparse_entries.shape)
    entries = np.asarray(sparse_entries.todense())
    """