import pandas as pd
import numpy as np
import pickle
import itertools
import time
import sys
from scipy.sparse.coo import coo_matrix

if sys.platform == 'darwin':

    PATH_TO_DATA = "/Users/Louis/PycharmProjects/policy_approximation/DATA/fedex_pc_cleaned_no_0.data"
    PATH_TO_ADDRESSES = "/Users/Louis/PycharmProjects/MEng_Research/foo-Environment_2/gym_foo/envs/addresses.fedex"
    PATH_TO_POSTAL_CODE = "/Users/Louis/PycharmProjects/policy_approximation/DATA/postal_codes_fedex"

elif sys.platform == 'linux':
    # TODO we didn't change the path as the data is available
    PATH_TO_DATA = "/Users/Louis/PycharmProjects/policy_approximation/DATA/fedex_pc_cleaned_no_0.data"
    PATH_TO_ADDRESSES = "/Users/Louis/PycharmProjects/MEng_Research/foo-Environment_2/gym_foo/envs/addresses.fedex"
    PATH_TO_POSTAL_CODE = "/Users/Louis/PycharmProjects/policy_approximation/DATA/postal_codes_fedex"



def postal_code_to_ID(postal_code):
    """
    Return the index (i.e. the ID) of postal_code
    :param postal_code:
    :return:
    """
    with open(PATH_TO_POSTAL_CODE, "rb") as f:
        pcodes = pickle.load(f)

    return pcodes.index(postal_code)  # index == ID !, unique if postal_codes.fedex correctly generated!

class Datasets:
    """
    Will allow to access, prepare fedex data in order to be fed into a neural net
    """

    def __init__(self, path, mode):

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
            column=mode,  # "postal_code" is not supported yet
            cur_loc_code=1,
            rem_deliv_code=0.5,
            rem_pickup_code=-0.5
        )

        indices = np.asarray(indices)
        rows = indices[:, 0]
        print("max rows indices ", max(rows))
        columns = indices[:, 1]
        print("max columns indices ", max(columns))
        sparse_entries = coo_matrix((values, (rows, columns)))
        print("sparse_entries shape ", sparse_entries.shape)
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

        :param range_tour: list, number of operation tours retrieved (1 tour = 1 truck operating)
            column: str, retrieve locations by Addresses or Postal Codes
            *_code: floats, score assigned in the generalized one-hot encoding

        :return: indices, values, dense_shape
        """
        beginning = time.time()
        # parameters to create a sparse tensor
        indices = []
        values = []

        indices_label = []
        values_label = []

        # encode postal codes
        with open(PATH_TO_POSTAL_CODE, "rb") as f:
            pcodes = pickle.load(f)  # pcodes[int] gives the postal code encoded by int

        post_code_to_int = dict((c, i) for c, i in zip(pcodes, range(0, len(pcodes))))

        array_line = 0
        print("Start retrieving sparse values")
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
                # TODO: for RNN, we need to keep record of the end of each delivery tour, because training
                # TODO will be done by tours!
                for step in range(end - start - 1):
                    if column == "address":
                        # cur loc
                        cur_loc = self.addresses_id_col[start + step]
                        # print("loc ",cur_loc)
                        # ...
                        # print("CUR LOC ", cur_loc)


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

                        # label
                        next_loc = self.addresses_id_col[start + step + 1]

                        indices_label.extend([[array_line, next_loc]])

                        values_label.extend([1])
                        array_line += 1
                        # print("LABEL ", next_loc)

                    elif column == "postal_code":
                        # cur loc
                        cur_loc = self.postal_code_col[start + step]
                        cur_loc = post_code_to_int[cur_loc]
                        # print("loc ",cur_loc)
                        # ...
                        # print("CUR LOC ", cur_loc)


                        # remaining deliveries
                        rem_deliv = list(self.get_rem_deliv([start + step + 1, end])["PostalCode"])
                        # print("rem_deliv ", rem_deliv)

                        # remaining pickups => maybe use POSTAL CODE instead
                        rem_pick = list(self.get_rem_pickup([start + step + 1, end])["PostalCode"])

                        # TODO ENCODE postal codes !! mapping from each postal code to an int (\in [1, ~16 000]
                        # TODO rem_pick and rem_deliv => encoded to their respective IDs
                        # translation
                        rem_deliv_id = [post_code_to_int[int(i)] for i in rem_deliv]
                        rem_pick_id = [post_code_to_int[int(i)] for i in rem_pick]

                        # add successively delivery, pickup, current location
                        # adapted to the SparseTensor requirements
                        indices.extend([[array_line, int(i)] for i in rem_deliv_id])
                        indices.extend([[array_line, int(i)] for i in rem_pick_id])
                        indices.extend([[array_line, cur_loc]])

                        values.extend([rem_deliv_code] * len(rem_deliv_id))
                        values.extend([rem_pickup_code] * len(rem_pick_id))
                        values.extend([cur_loc_code])

                        # label
                        next_loc = self.postal_code_col[start + step + 1]
                        next_loc = post_code_to_int[next_loc]  # /!\ Encoding /!\

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
        if sys.platform == 'darwin':
            loaded = np.load('DATA/dataset_pc.fedex.npz')  # dataset_addresses.fedex.npz
        elif sys.platform == 'linux':
            loaded = np.load('/home/louis/Documents/Research/policy_approximation-master/DATA/dataset_pc.fedex.npz')
        self.entries = loaded['inputs']
        self.labels = loaded['labels']

        print(" DATA SUCCESSFULLY LOADED ! ")
        print("Shape of the entries:    ", self.entries.shape)
        print("Shape of the labels:    ", self.labels.shape)
        print(" ")

        self.num_examples = self.entries.shape[0]
        self.num_features = self.entries.shape[1]

        print("____________________________________________________________________________")


class Dataset:

    def __init__(self, entries, labels):
        """

        :param percentage: percentage, float. Proportion of dataset used for training
        """

        # to follow training
        self.index_in_epoch = 0
        self.epochs_completed = 0

        # TRAINING DATASET
        self.entries = entries
        self.labels = labels

        # Sparse version
        sp_data = coo_matrix(self.entries)

        self.idc = (sp_data.row, sp_data.col)
        self.val = sp_data.data
        self.shape = sp_data.shape

        # data shape
        self.num_examples = entries.shape[0]
        self.num_features = entries.shape[1]

    # TODO: this is the feeding technique. Other are possible (queues with list of files)
    # TODO: we will have to include an option for RNN training (train using meaningful sequences of examples
    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from entries and labels.

        :param batch_size, int.
        :param shuffle, bool. shuffle the next batch

        :return tuple, np.array entries AND np.array labels
        """
        start_ = self.index_in_epoch

        # Shuffle for the first epoch

        if self.epochs_completed == 0 and start_ == 0 and shuffle:
            perm0 = np.arange(self.num_examples)
            np.random.shuffle(perm0)
            self.entries = self.entries[perm0, :]
            self.labels = self.labels[perm0]

        # Go to the next epoch
        if start_ + batch_size > self.num_examples:
            # Finished epoch
            self.epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self.num_examples - start_
            entries_rest_part = self.entries[start_:self.num_examples, :]
            labels_rest_part = self.labels[start_:self.num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self.num_examples)
                np.random.shuffle(perm)
                self.entries = self.entries[perm, :]
                self.labels = self.labels[perm]
            # Start next epoch
            start_ = 0
            self.index_in_epoch = batch_size - rest_num_examples
            end = self.index_in_epoch
            entries_new_part = self.entries[start_:end, :]
            labels_new_part = self.labels[start_:end]
            result = np.concatenate((entries_rest_part, entries_new_part),
                                    axis=0),\
                     np.concatenate((labels_rest_part, labels_new_part),
                                    axis=0)
            return result

        else:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            return self.entries[start_:end, :], self.labels[start_:end]

    def next_sp_batch(self, batch_size, shuffle=True):
        """
        Equivalent of self.next_batch but for sparse data
        :param batch_size:
        :param shuffle:
        :return: indices, values, shape, labels
        """
        entries, labels = self.next_batch(batch_size=batch_size, shuffle=shuffle)

        # sparse transformation
        # TODO for full speed up, sparse matrix should be written in files and access as such
        # Gain: 10mn every 100,000 step (not a priority though)
        sp_batch = coo_matrix(entries)


        return np.column_stack((np.asarray(sp_batch.row), np.asarray(sp_batch.col))), sp_batch.data, sp_batch.shape,\
               labels

if __name__ == "__main__":

    # this is looong
    fedex = Datasets(PATH_TO_DATA, mode='postal_code')
    np.savez_compressed("dataset_pc.fedex", inputs=fedex.entries, labels=fedex.labels)

    """
    raw_data = pd.read_csv(
        PATH_TO_DATA,
        header=0,
        delim_whitespace=True
    )

    postal_codes = list(sorted(set(list(raw_data["PostalCode"]))))
    print(max(raw_data["PostalCode"]))

    with open(PATH_TO_POSTAL_CODE, 'rb') as f:
        pcodes = pickle.load(f)

        print(len(pcodes))
    """

