# import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import itertools

# FedEx datapath
# CSV
raw_data_path = "/Users/Louis/PycharmProjects/MEng_Research/foo-Environment_2/dynamics/demand_models/fedex.data"
DATA_FRAME = pd.read_csv(raw_data_path, header=0, delim_whitespace=True)

NUMBER_ENTRIES = 33  # len(DATA_FRAME["PostalCode"])  # 116 408
NUMBER_LOCATIONS = int(DATA_FRAME.loc[DATA_FRAME['Address'].idxmax()]["Address"])
NUMBER_POSTAL_CODE = len(set(list(DATA_FRAME["PostalCode"])))

END_TIME = int(DATA_FRAME.loc[DATA_FRAME['StopStartTime'].idxmax()]["StopStartTime"])
START_TIME = int(DATA_FRAME.loc[DATA_FRAME['StopStartTime'].idxmin()]["StopStartTime"])

print("start ", START_TIME)
print("end ", END_TIME)

"""
## TENSORFLOW declaration
time_boundaries = list(np.linspace(
    int_to_second(START_TIME),
    int_to_second(END_TIME),
    100
))

# time : using encoding in seconds
current_time = tf.contrib.layers.real_valued_column("current_time")
cur_time_bucket = tf.contrib.layers.bucketized_column(current_time,
                                                      boundaries=time_boundaries
                                                      )
# loc
current_location = tf.contrib.layers.sparse_column_with_hash_bucket("current_location",
                                                                    hash_bucket_size=NUMBER_LOCATIONS
                                                                    )
# deliv
remaining_deliveries = tf.contrib.layers.sparse_column_with_hash_bucket("remaining_deliveries",
                                                                        hash_bucket_size=NUMBER_LOCATIONS
                                                                        )
# pickup
remaining_pickups = tf.contrib.layers.sparse_column_with_hash_bucket("remaining_pickups",
                                                                     hash_bucket_size=NUMBER_LOCATIONS
                                                                     )
"""
def build_estimator():
    """

    :param :
    :return:
    """
    # continuous column
    current_time = tf.contrib.layers.real_valued_column("current_time")

    # TODO this seems really bad, as each input data will be seen as a category...
    # sparse column for locations
    locations = tf.contrib.layers.sparse_column_with_hash_bucket(
        "locations",
        hash_bucket_size=int(1e5)
    )

    # transformation
    print("start_time ", int_to_second(START_TIME))
    print("end_time ", int_to_second(END_TIME))

    time_boundaries = list(np.linspace(
        int_to_second(START_TIME),
        int_to_second(END_TIME),
        100
    ))
    cur_time_bucket = tf.contrib.layers.bucketized_column(current_time,
                                                          boundaries=time_boundaries
                                                          )


    # wide columns
    wide_columns = [cur_time_bucket,
                    locations # ,
                    # tf.contrib.layers.crossed_column([locations, current_time], hash_bucket_size=int(1e6))
                    ]

    m = tf.contrib.learn.LinearClassifier(
        feature_columns=wide_columns
    )

    return m


def input_fn(data_frame):

    # Creates a dictionary mapping from each continuous feature column name (k) to
    # the values of that column stored in a constant Tensor.

    # DATA PREPARATION
    # parameters to create a sparse tensor
    indices = []
    values = []
    dense_shape = [NUMBER_ENTRIES-1, NUMBER_LOCATIONS]

    indices_label = []
    values_label = []

    # creating trips set
    trip_start_index = list(data_frame[data_frame["StopOrder"] == 1 ]["StopOrder"].index)
    up_idx = np.asarray(trip_start_index[1:])
    down_idx = np.asarray(trip_start_index[0:-1])
    # trip_step_nb = up_idx - down_idx

    for start, end in itertools.zip_longest(trip_start_index[0:2], trip_start_index[1:2], fillvalue=0):
        # start: beginning of truck's trip
        # end: ...

        # print('start ', start)
        # print('end', end)

        if end != 0:
            for step in range(end - start-1):
                # time
                cur_time = data_frame["StopStartTime"][start + step]
                # print("time ", cur_time)
                # ... (+ week day?)
                # print("CUR TIME ", cur_time)

                # cur loc
                cur_loc = data_frame["Address"][start + step]
                # print("loc ",cur_loc)
                # ...
                # print("CUR LOC ", cur_loc)

                # remaining deliveries => maybe use POSTAL CODE
                subdata_frame = data_frame[start + step +1:end]
                rem_deliv = list(subdata_frame[subdata_frame["PickupType"] == 0]["Address"].values)
                # print("rem_deliv ", rem_deliv)

                # remaining pickups => maybe use POSTAL CODE instead
                rem_pick = list(subdata_frame[subdata_frame["PickupType"] != 0]["Address"].values)
                # print("rem_pick ", rem_pick)

                # label
                next_loc = data_frame["Address"][start + step + 1]

                indices_label.extend([[start + step, next_loc]])
                values_label.extend([1])
                # print("LABEL ", next_loc)

                # add successively delivery, pickup, current location
                # adapted to the SparseTensor requirements
                indices.extend([[start + step, i] for i in rem_deliv])
                indices.extend([[start + step, i] for i in rem_pick])
                indices.extend([[start + step, cur_loc]])

                values.extend([-0.25]*len(rem_deliv))
                values.extend([-0.5]*len(rem_pick))
                values.extend([0.5])


    print("index ", type(indices))
    print("values ", type(values))

    # dictionary mappings
    print("cur_time_type ", list(data_frame["StopStartTime"][:-1].apply(lambda x: int_to_second(x)).astype(float)))
    continuous_col = {
        "current_time":tf.constant(
            list(data_frame["StopStartTime"][:-1].apply(lambda x: int_to_second(x)).astype(float)),
            dtype=tf.float16
        )
    }
    categorical_col = {
        "locations": tf.SparseTensor(
            indices=indices,
            values=values,
            dense_shape=np.asarray(dense_shape)
        )
    }

    # merge the 2 dictionaries
    feature_cols = dict(continuous_col)
    feature_cols.update(categorical_col)

    # label column
    label = tf.SparseTensor(
        indices=indices_label,
        values=values_label,
        dense_shape=np.asarray(dense_shape)
    )
    print("test")

    return feature_cols, label


# ------------- FUNCTIONS ----------------
def address_viewer(int, data_frame):
    """
    Return informations about an address ID (int)

    :param int: index in the data_frame["Address"] column
            data_frame, pandas dataframe with an Address, Longitude and Latitude column
    :return: address (string), addressID, Lat, Long
    """
    addresses_path = "/Users/Louis/PycharmProjects/MEng_Research/foo-Environment_2/gym_foo/envs/addresses.fedex"
    with open(addresses_path, "rb") as f:
        addresses = pickle.load(f)

    return addresses[data_frame["Address"][int]], data_frame["Address"][int], data_frame["Longitude"][int], data_frame["Latitude"][int]

def int_to_second(time):
    """
    Return number of seconds since 00:00 for an input int (1020 -> 10h20)

    :param int:
    :return:
    """

    m = time % 100
    h = int((time - m)/100)
    return h * 3600 + m * 60


if __name__ == "__main__":

    # train param
    model_dir = "/Users/Louis/tensorflow"
    train_steps = 1
    idx_start = list(DATA_FRAME[DATA_FRAME["StopOrder"] == 1]["StopOrder"].index)
    print(len(idx_start))
    # train set
    df_train = DATA_FRAME[:idx_start[-300]]
    # test set
    df_test = DATA_FRAME[idx_start[300]:]

    m = build_estimator()
    m.fit(input_fn=lambda: input_fn(DATA_FRAME), steps=train_steps)
    # results = m.evaluate(input_fn=lambda: input_fn(df_test), steps=1)
    # for key in sorted(results):
        # print("%s: %s" % (key, results[key]))
