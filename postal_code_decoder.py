import pandas as pd
import numpy as np
import pickle
import geocoder
import matplotlib.pyplot as plt
import itertools
import time
import math
from scipy.sparse.coo import coo_matrix

PATH_TO_DATA = "/Users/Louis/PycharmProjects/MEng_Research/foo-Environment_2/dynamics/demand_models/fedex.data"
PATH_TO_ADDRESSES = "/Users/Louis/PycharmProjects/MEng_Research/foo-Environment_2/gym_foo/envs/addresses.fedex"


def ID_to_address(id):
    addresses_path = "/Users/Louis/PycharmProjects/MEng_Research/foo-Environment_2/gym_foo/envs/addresses.fedex"
    with open(addresses_path, "rb") as f:
        addresses = pickle.load(f)

    return addresses[id]


def cleaner(raw_data):
    # among wrong format only 14 postal codes are non 0

    raw_data_copy = raw_data
    # df of wrong postal code
    wrong_postal_codes = raw_data[~raw_data["PostalCode"].isin(range(10000, 830000))]["PostalCode"]
    wrong_postal_codes_idx = wrong_postal_codes.index

    # list of corresponding addresses
    add_of_wrong_postal_codes = list(raw_data[~raw_data["PostalCode"].isin(range(10000, 830000))]["Address"])

    print("indices of wrong postal code ", len(wrong_postal_codes.index))
    print("wrong postal codes", wrong_postal_codes)
    print("address of wrong postal code ", add_of_wrong_postal_codes)

    # retrieve postal codes
    successful_pc_retrieval = 0
    list_retrieved_pc = []

    for add in add_of_wrong_postal_codes:
        try:
            correct_postal_code = int(geocoder.google(ID_to_address(add)+", Singapore").postal)
            if correct_postal_code in range(10000, 830000):
                list_retrieved_pc.append(correct_postal_code)
                successful_pc_retrieval += 1
                print("Successfully retrieved a postal code")
            else:
                print("postal code not in the range, strange for google !")
                list_retrieved_pc.append(0)

        except: # error in retrieving
            print("An error occured on geocoder")
            list_retrieved_pc.append(0)

    print("Successfully retrieved postal code : ", successful_pc_retrieval)

    if len(list_retrieved_pc) == len(wrong_postal_codes):
        # replacing into dataframe
        raw_data_copy.loc[wrong_postal_codes_idx, 'PostalCode'] = list_retrieved_pc

        # writing pandas -> numpy array -> csv (to have the correct data type)
        array = np.asarray(raw_data_copy.as_matrix())
        header = "StopDate,WeekDay,StopOrder,StopStartTime,Address,PostalCode,\
            CourierSuppliedAddress,ReadyTimePickup,CloseTimePickup,PickupType,\
            WrongDayLateCount,RightDayLateCount,FedExID,Longitude,Latitude"

        fmt = ['%.8d', '%.2d', '%.2d', '%.4d', '%.6d', '%.4d', '%.4d', '%.4d', '%.4d', '%.4d', '%.4d', '%.4d',
               '%.4d', '%.18f','%.18f']

        np.savetxt('fedex_pc_cleaned.data', array, fmt=fmt, header=header)
        print("File written, hope it's fine !")

    else:
        print("Error, nb of attempted retrieved pc do not match nb of wrong postal code")
        return 0

if __name__ == "__main__":
    PATH_TO_POSTAL_CODE = "/Users/Louis/PycharmProjects/policy_approximation/DATA/postal_codes_2_fedex"

    """
    # this is looong
    # fedex = Datasets(PATH_TO_DATA)
    # np.savez_compressed("dataset.fedex", inputs=fedex.entries, labels=fedex.labels)


    # Cleaner of data (to be runned again)
    raw_data = pd.read_csv(
        PATH_TO_DATA,
        header=0,
        delim_whitespace=True
    )

    cleaner(raw_data=raw_data)
    """
    """
    print("_____________________________________")
    path = "/Users/Louis/PycharmProjects/policy_approximation/DATA/fedex_pc_cleaned.data"
    data_cleaned = pd.read_csv(
        path,
        header=0,
        delim_whitespace=True
    )

    data_cleaned_copy = data_cleaned
    print("No postal code ", len(data_cleaned[data_cleaned["PostalCode"] == 0])/len(data_cleaned))
    postal_code_to_sample_from = list(data_cleaned[data_cleaned["PostalCode"] != 0]["PostalCode"])
    idx_error_input = data_cleaned[data_cleaned["PostalCode"] == 0].index

    fill_postal_code = np.random.choice(postal_code_to_sample_from, len(idx_error_input))
    print(fill_postal_code)

    data_cleaned_copy.loc[idx_error_input, 'PostalCode'] = fill_postal_code

    print("are there still 0 postal codes? ", len(data_cleaned_copy[data_cleaned_copy["PostalCode"] == 0 ]))

    array = np.asarray(data_cleaned_copy.as_matrix())
    header = "StopDate WeekDay StopOrder StopStartTime Address PostalCode CourierSuppliedAddress ReadyTimePickup" \
             " CloseTimePickup PickupType WrongDayLateCount RightDayLateCount FedExID Longitude Latitude"

    fmt = ['%.8d', '%.2d', '%.2d', '%.4d', '%.6d', '%.4d', '%.4d', '%.4d', '%.4d', '%.4d', '%.4d', '%.4d',
           '%.4d', '%.18f', '%.18f']

    np.savetxt('fedex_pc_cleaned_no_0.data', array, fmt=fmt, header=header)
    print("File written, hope it's fine !")

    # list creation to replace wrong postal codes
    """
    path = "/Users/Louis/PycharmProjects/policy_approximation/DATA/fedex_pc_cleaned_no_0.data"
    data_cleaned = pd.read_csv(
        path,
        header=0,
        delim_whitespace=True
    )

    unique_postal_code = sorted(list(set(data_cleaned["PostalCode"])))
    print("allo", unique_postal_code)


    with open("DATA/postal_codes_2_fedex", "rb") as f:
        pc = pickle.load(f)

    print(pc.index(18000))
