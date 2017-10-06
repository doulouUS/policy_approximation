import pandas as pd
import numpy as np
import pickle
import itertools
import time
# import sys

from scipy.sparse.coo import coo_matrix
# from matplotlib import pyplot
# import matplotlib as mpl
import matplotlib.pyplot as plt

# Mapping
from bokeh.io import output_file, show
from bokeh.models import (
  GMapPlot, GMapOptions, ColumnDataSource, Circle, DataRange1d, PanTool, WheelZoomTool, BoxSelectTool, Line, LabelSet
)
from datetime import datetime, timedelta
# from bokeh.palettes import inferno

# import tensorflow as tf
# from tensorflow.python.framework import graph_util
# from scipy.spatial.distance import pdist

# Homemade modules
import sys
sys.path.insert(0, '/Users/Louis/PycharmProjects/policy_approximation')
# import tf_model_handling_toolbox as tf_model

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
        int_to_post_code = pickle.load(f)

    return int_to_post_code.index(postal_code)  # index == ID !, unique if postal_codes.fedex correctly generated!

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
            column=mode,
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
            int_to_post_code = pickle.load(f)  # int_to_post_code[int] gives the postal code encoded by int

        post_code_to_int = dict((c, i) for c, i in zip(int_to_post_code, range(0, len(int_to_post_code))))

        array_line = 0  # keep track of the row position in the raw data
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
    "Fast access to pre-saved data"

    def __init__(self, mode="dense"):
        # TODO implement the "sparse" mode
        #  - store indices and values by batches (one batch = one line of the dense array) using python dict?
        #  -
        """

        :param mode: string, "dense" (for entries and labels) or "sparse" (for indices, values and labels)
        :param specific_entries, list of int, specifies which rows to take from the whole data
        """

        print("____________________________________________________________________________")
        print(" ")
        print(" DATA LOADING...")

        if mode == "dense":
            if sys.platform == 'darwin':
                loaded = np.load('DATA/dataset_pc.fedex.npz')
            elif sys.platform == 'linux':
                loaded = np.load('/home/louis/Documents/Research/policy_approximation-master/DATA/dataset_pc.fedex.npz')

            self.entries = loaded['inputs']
            self.labels = loaded['labels']

        elif mode == "sparse":
            # TODO correct file names
            if sys.platform == 'darwin':
                loaded = np.load('DATA/dataset_pc.fedex.npz')
            elif sys.platform == 'linux':
                loaded = np.load('/home/louis/Documents/Research/policy_approximation-master/DATA/dataset_pc.fedex.npz')

            # self.indices = ...
        else:
            print('mode param is wrong. Either dense or sparse, here it is set to ', mode)


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

        :param entries, np array samples*features
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

        return np.column_stack((np.asarray(sp_batch.row), np.asarray(sp_batch.col))),\
               sp_batch.data,\
               sp_batch.shape,\
               labels


# BOKEH MAP
def initialize_map(title="Singapore", lat=1.29, lng=103.8, maptype="roadmap", zoom=11):
    """
    Bokeh map initialization for visualization

    :param title: str
    :param lat: float
    :param lng: float
    :param maptype: str
    :param zoom: int
    :return: plot
    """
    map_options = GMapOptions(lat=lat, lng=lng, map_type=maptype, zoom=zoom)

    plot = GMapPlot(
        x_range=DataRange1d(), y_range=DataRange1d(), map_options=map_options
    )
    plot.title.text = title
    plot.api_key = "AIzaSyApMtNEiMXlpRju4TR8my3lK_0tG-VafPU"

    return plot


def add_point_on_map(lat, lng, plot, label, color="blue"):
    """
    Plot lat and long
    :param lat: list of float
    :param lng: list of float, same size as lat
    :param plot: GMapPlot object, from initialize_map function
    :param str, label
    :return:
    """
    if len(lat) != len(lng):
        raise ValueError('latitude and longitude lists should be of same size.')

    source = ColumnDataSource(
        data=dict(
            lat=lat,
            lon=lng,
            label=[label]
        )
    )
    #
    circle = Circle(x="lon", y="lat", size=8, fill_color=color, fill_alpha=0.8, line_color=None)
    plot.add_glyph(source, circle)

    # Label
    labels = LabelSet(x='lat', y='lon', level='glyph', text='label',
                      x_offset=5, y_offset=5, source=source, render_mode='canvas')
    plot.add_layout(labels)


def add_line_on_map(lat, lng, plot, label, color="black"):
    """
    Plot lat and long
    :param lat: list of float
    :param lng: list of float, same size as lat
    :param plot: GMapPlot object, from initialize_map function
    :param label: list of str, label the line

    :return:
    """
    # TODO add label to line
    if len(lat) != len(lng):
        raise ValueError('latitude and longitude lists should be of same size.')

    source = ColumnDataSource(
        data=dict(
            lat=lat,
            lon=lng
        )
    )
    # Glyph
    line = Line(x="lon", y="lat", line_color=color)
    plot.add_glyph(source, line)



def show_map(plot):
    """
    Show map with widgets, possible to generate html
    :param plot: GMapPlot object, from initialize_map function
    :return:
    """
    # Add widgets
    plot.add_tools(PanTool(), WheelZoomTool(), BoxSelectTool())
    # output_file("gmap_plot.html")
    show(plot)

def basic_settings(nb_truck=5):
    """
    Extract data of interest from the bulk of FedEx data

    :param nb_truck: int, nb of trucks to use as a source of data, taking the most numerous
    :return: data, pd dataframe: contains all the info we're interested in
    :return: data_indices, list of int: indices to find the data in the whole dataset
    :return: sorted_truck, list of int: IDs of FedEx trucks sorted by number of tasks
    :return: idx_to_pc, pc_to_idx: dicts, for having a unique reference for each postal code
    """
    raw_data = pd.read_csv(
        PATH_TO_DATA,
        header=0,
        delim_whitespace=True
    )
    # FIRST CANDIDATE: /!\ Pb it is only deliveries /!\
    # Truck               976220
    # 1 region: ["Latitude"] < 1.3463879115 and ["Longitude"] > 103.8965034485
    # 554 postal code in this region
    # 901 samples
    # 12 tours in a month

    # SECOND CANDIDATE
    # Truck        868386
    # 1 region  ["Latitude"] < 1.31216860 and ["Longitude"] > 103.78200827 and ["Longitude"] < 103.8705063913
    # postal codes?
    # 757 samples - 395 deliveries
    # 37 postal codes
    # 11 tours

    #
    # Ranking of trucks having the most jobs before 12pm
    trucks_id = np.asarray(list(set(raw_data["FedExID"])))

    nb_loc = []
    for truck in trucks_id:
        # df1.loc[lambda df: df.A > 0, :]
        truck_nb_cust = raw_data[raw_data["FedExID"] == truck]
        add = len(truck_nb_cust[truck_nb_cust["StopStartTime"] < 1200]["PostalCode"])
        nb_loc.append(int(add))

    idx_truck = np.argsort(-np.asarray(nb_loc))
    sorted_truck = trucks_id[idx_truck]
    # 147888 5034428  826678  792262 5029354  774331 5015037  976220
    # 305366 5007991  861864 5034416 5025403  214484 5025441  775391  938771
    # 912979  297524  735762 5022223  868386 5002569  986246  826444 5057185
    # 909334  363777  955144  218158  940061  363851  955141  244089 5081861
    # 982516  436132  955131  151454  968292  616349  814086  767395  299173

    # TODO: select only <12pm jobs + more data !
    data = raw_data.loc[raw_data["FedExID"].isin(sorted_truck[0:nb_truck])]
    # Further selection
    # data = raw_data[raw_data["FedExID"] == 868386]
    # data = data[data["Latitude"] < 1.31216860]
    # data = data[data["Longitude"] > 103.78200827]
    # data = data[data["Longitude"] < 103.8705063913]
    data = data[data["StopStartTime"] < 1200]
    data_indices = list(data.index)
    print("length ", len(data_indices))

    # postal code encoding
    idx_to_pc = sorted(list(set(data["PostalCode"])))
    pc_to_idx = {j: i for i, j in enumerate(idx_to_pc)}

    # TODO identify scenario used for the simulator: by truck, day
    # Truck 5029354, first day 1/12/2015
    # truck_id = sorted_truck[4]
    data_scenario = data[data["FedExID"] == 5029354]

    # select different days
    # day_scenario = list(set(data_scenario["StopDate"]))[0]
    day_scenario = data_scenario["StopDate"].iloc[0]

    print(" ----- SCENARIO LOADED ----- ")
    print("Truck considered         : ", 5029354)  # truck_id)
    print("Days to choose from      : ", set(data_scenario["StopDate"]))
    print("Day re-created           : ", day_scenario)
    print(" ")
    # print("day chosen for the simulator: ", day_scenario)
    data_scenario = data_scenario[data_scenario["StopDate"] == day_scenario]
    print("Number of locations used for training, L=", len(idx_to_pc))

    return data, data_indices, data_scenario, sorted_truck, idx_to_pc, pc_to_idx


def data_prep_2nd_attempt(
                          nb_truck=5,
                          method="classification",
                          deliv_reward = 1,
                          pickup_reward = 0.5):
    """

    :param idx_to_be_removed: list of int, correspond to the indices used to build the scenario in the simulator
    (optional)
    :param nb_truck: int, nb of trucks to consider among the ones having the most jobs
    :param method: str, "classification" or "regression"
    :param deliv_reward: float
    :param pickup_reward: float
    :return:
    """

    # retrieve basic data
    data_init, data_indices, data_scenario, sorted_truck, idx_to_pc, pc_to_idx = basic_settings(nb_truck=nb_truck)
    print("Before filtering shape ", data_init.shape)

    # TODO BIG PROBLEM: by filtering just this day of operation for one specific truck, we remove the number of labels
    # in the training samples. Thus, we have to include in our training even the scenario used in the simulator...
    # This is because we lack data: we need more samples per location (here one job/location almost...)
    # Filter data from data_scenario
    # data = data_init.ix[list(set(data_init.index) - set(data_scenario.index))]
    data = data_init

    # Data to be built
    indices = []
    values = []
    shape_count = 0

    indices_label = []
    values_label = []

    # Find tours index
    start_tour_index = []
    end_tour_index = []

    start_tour_index.append(data_indices[0])
    for i, j in zip(data_indices, range(0, len(data_indices)-1)):
        if data_indices[j+1] - i-1 > 10:
            start_tour_index.append(data_indices[j + 1])
            end_tour_index.append(i)
    end_tour_index.append(data_indices[-1])

    # retrieve for each entry, known jobs remaining to be done at the time of the entry in the current tour
    for index, row in data.iterrows():
        if index not in end_tour_index:

            # current start and end tour index
            tour_end = min(filter(lambda x: x > index, end_tour_index))
            tour_start = max(filter(lambda x: x <= index, start_tour_index))

            ## Gathering the state
            # remaining jobs in the tour
            data_indices_tour = [i for i in data.index if index < i <= tour_end]

            # remaining deliv
            rem = data.loc[data_indices_tour]
            rem_deliv = list(rem[rem["ReadyTimePickup"] == 0000]["PostalCode"])

            # remaining pickups, known at the current time
            rem_pick = rem[rem["ReadyTimePickup"] != 0000]
            rem_pick = list(rem_pick[rem_pick["ReadyTimePickup"] <= row["StopStartTime"]]["PostalCode"])

            # current location
            cur_loc = row["PostalCode"]

            # current time encoded as nb of seconds between 12pm and now/nb seconds between 12pm and 12am
            total_nb_seconds = timedelta(hours=12, minutes=0)
            cur_time = datetime.strptime(str(int(row["StopStartTime"])), '%H%M').time()  # time object
            cur_time = timedelta(hours=cur_time.hour, minutes=cur_time.minute)  # nb of seconds from 12am
            # TODO this can further be noramlized as most values will be >0 (>6am)
            cur_time = 2*cur_time.seconds / total_nb_seconds.seconds - 1  # normalized time in [-1,1]

            # decision (next row)
            next_loc = data.loc[data_indices_tour[0]]
            next_loc = next_loc["PostalCode"]

            # encode locations
            rem_deliv = [pc_to_idx[idx] for idx in rem_deliv]
            rem_pick = [pc_to_idx[idx] for idx in rem_pick]
            cur_loc = pc_to_idx[cur_loc]
            next_loc = pc_to_idx[next_loc]

            # build sparse elements
            # TODO be cautious that there is one element of your input vector that is the current time
            # indices
            indices.extend([[shape_count, i + 1] for i in rem_deliv])
            indices.extend([[shape_count, i + 1] for i in rem_pick])
            indices.extend([[shape_count, 0]])
            indices.extend([[shape_count, cur_loc + 1]])


            values.extend([0.5]*len(rem_deliv))
            values.extend([-0.5]*len(rem_pick))
            values.extend([cur_time])
            values.extend([1])

            # So that the tour's first locations appears in the dataset as labels
            if index in start_tour_index:
                # First location of a tour: add the depot as
                indices.extend([[shape_count+1, i + 1] for i in rem_deliv])
                indices.extend([[shape_count+1, i + 1] for i in rem_pick])
                indices.extend([[shape_count+1, 0]])
                indices.extend([[shape_count+1, cur_loc + 1]])

                values.extend([0.5] * len(rem_deliv))
                values.extend([-0.5] * len(rem_pick))
                values.extend([0.3])  # departure time less than 8am
                values.extend([0.5])  # second way to differentiate this entry as the first of the tour


            if method == "classification":
                # labels
                indices_label.extend([[shape_count, next_loc]])
                values_label.extend([1])

                if index in start_tour_index:
                    indices_label.extend([[shape_count+1, cur_loc]])
                    values_label.extend([1])

                    shape_count += 2

                else:
                    shape_count += 1

            if method == "regression":
                # label
                # jobs done so far in the tour
                job_done_indices = [i for i in data.index if tour_start <= i <= index]

                # nb of deliv done
                done = data.loc[job_done_indices]
                done_deliv = len(list(done[done["ReadyTimePickup"] == 0000]["PostalCode"]))

                # nb of pickup done
                done_pick = len(list(done[done["ReadyTimePickup"] != 0000]["PostalCode"]))

                # total score
                score = done_deliv * deliv_reward + done_pick * pickup_reward

                # add label
                values_label.extend([score])

                shape_count += 1

    shape = [shape_count, len(idx_to_pc) + 1]  # + 1 because there is the time dimension !
    print("Shape of the data", shape)
    if method == "classification":
        return  values, indices, values_label, indices_label, shape
    elif method == "regression":
        return values, indices, values_label, shape


if __name__ == "__main__":
    # values, indices, values_label, shape = data_prep_2nd_attempt(method="regression")
    # print("values", len(values_label))

    basic_settings()
    # data_prep_2nd_attempt()

    """
    data_filtered, _, _, _, _, _ = data_prep_2nd_attempt(nb_truck=15)

    locations = list(set(list(data_filtered["PostalCode"])))
    data_filtered = data_filtered[data_filtered["ReadyTimePickup"] != 0]

    print("shape ", data_filtered.shape)
    print("count ", data_filtered[data_filtered["PostalCode"] == 456715]["ReadyTimePickup"])
    pickup_arrival_time = {
        location: list(data_filtered[data_filtered["PostalCode"] == location]["ReadyTimePickup"])
        for location in locations
        }

    print(pickup_arrival_time.items())

    index = []
    data = []
    for key, val in pickup_arrival_time.items():
        # plot only locations where we have more info
        if len(val) > 10:
            index.append(key)
            data.append(val)

    fig, ax = plt.subplots(ncols=1)
    ax.boxplot(data)
    ax.set_xticklabels(index)
    # ax2.violinplot(data)
    # ax2.set_xticks(range(1, len(index) + 1))
    # ax2.set_xticklabels(index)

    plt.show()

    values, indices, values_label, indices_label, shape = data_prep_2nd_attempt()
    indices = np.asarray(indices)
    rows = indices[:, 0]
    columns = indices[:, 1]
    print("Nb of columns ", len(columns))
    print("Nb of rows ", len(rows))
    dense_data = coo_matrix(values, (rows, columns)).todense()

    # represent extract
    # image = np.reshape(dense_data[0,:], (69,233))


    data = ReadDataFedex()
    data = Dataset(data.entries, data.labels)
    data_to_image, _ = data.next_batch(1, shuffle=True)
    print("Shape data_to_image ", type(data_to_image))
    image = np.reshape(data_to_image, (69,233))

    # print this image...
    # make a color map of fixed colors
    cmap = mpl.colors.ListedColormap(['blue', 'black', 'red', 'orange'])
    bounds = [-0.5, -0.3, 0.3, 0.8, 1.2]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    # tell imshow about color map so that only set colors are used
    print("ee", image.shape)
    img = pyplot.imshow(image, interpolation='nearest',
                        cmap=cmap, norm=norm)

    # make a color bar
    pyplot.colorbar(img, cmap=cmap,
                    norm=norm, boundaries=bounds, ticks=[-.5, 0, 1])

    pyplot.show()



    # Data of interest
    raw_data = pd.read_csv(
        PATH_TO_DATA,
        header=0,
        delim_whitespace=True
    )
    raw_data = raw_data[raw_data["FedExID"] == 868386]

    # Encoder and decoder for postal codes
    with open("/Users/Louis/PycharmProjects/policy_approximation/DATA/postal_codes_fedex", 'rb') as f:
        id_to_pc = pickle.load(f)

    pc_to_id = {j: i for i, j in enumerate(id_to_pc)}

    # Embeddings
    # We import the meta graph and retrieve a Saver
    input_checkpoint = "/Users/Louis/PycharmProjects/policy_approximation/DATA/embeddings_validation/" \
                       "log_10dim_10btch/model.ckpt-200000"

    output_node_names = ["embeddings_de_dingue_postal_codes"]
    embeddings = tf_model.retrieve_matrix(input_checkpoint, output_node_names[0])

    # Find tours index
    start_tour_index = []
    end_tour_index = []

    # ----------------------------- Index of the starting tours
    data_indices = list(raw_data.index)
    start_tour_index.append(data_indices[0])
    for i, j in zip(data_indices, range(0, len(data_indices)-1)):
        if data_indices[j+1] - i-1 > 10:
            start_tour_index.append(data_indices[j+1])
            end_tour_index.append(i)

    end_tour_index.append(data_indices[-1])
    print(start_tour_index)
    tour_indices = [i for i in raw_data.index if start_tour_index[0] < i < end_tour_index[0] + 1]

    # MAP making
    plot = initialize_map()
    colors = inferno(len(end_tour_index))
    add_point_on_map(lat=raw_data.loc[tour_indices]['Latitude'],
                     lng=raw_data.loc[tour_indices]['Longitude'],
                     plot=plot
                     )
    add_line_on_map(lat=raw_data.loc[tour_indices]['Latitude'],
                    lng=raw_data.loc[tour_indices]['Longitude'],
                    plot=plot,
                    color=colors[0]
                    )

    # Reveal on browser
    show_map(plot)
    """
