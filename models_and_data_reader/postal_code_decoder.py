import pandas as pd
import numpy as np
import pickle
# import geocoder
# from read_data_fedex import visualize_postal_code as viz

PATH_TO_DATA = "/Users/Louis/PycharmProjects/MEng_Research/foo-Environment_2/dynamics/demand_models/fedex.data"
PATH_TO_ADDRESSES = "/Users/Louis/PycharmProjects/MEng_Research/foo-Environment_2/gym_foo/envs/addresses.fedex"

# API Google
API_key = "-----------------------------------------------"
url = "https://maps.googleapis.com/maps/api/geocode/json?"


def ID_to_address(id):
    addresses_path = "/Users/Louis/PycharmProjects/MEng_Research/foo-Environment_2/gym_foo/envs/addresses.fedex"
    with open(addresses_path, "rb") as f:
        addresses = pickle.load(f)

    return addresses[id]

def clutter():
    """
    This is not really a function, just the code used in the main below, to clean the data
    :return:
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

    print("_____________________________________")
    path = "/Users/Louis/PycharmProjects/policy_approximation/DATA/fedex_pc_cleaned.data"
    data_cleaned = pd.read_csv(
        path,
        header=0,
        delim_whitespace=True
    )

    data_cleaned_copy = data_cleaned
    print("No postal code ", len(data_cleaned[data_cleaned["PostalCode"] == 0]) / len(data_cleaned))
    postal_code_to_sample_from = list(data_cleaned[data_cleaned["PostalCode"] != 0]["PostalCode"])
    idx_error_input = data_cleaned[data_cleaned["PostalCode"] == 0].index

    fill_postal_code = np.random.choice(postal_code_to_sample_from, len(idx_error_input))
    print(fill_postal_code)

    data_cleaned_copy.loc[idx_error_input, 'PostalCode'] = fill_postal_code

    print("are there still 0 postal codes? ", len(data_cleaned_copy[data_cleaned_copy["PostalCode"] == 0]))

    array = np.asarray(data_cleaned_copy.as_matrix())
    header = "StopDate WeekDay StopOrder StopStartTime Address PostalCode CourierSuppliedAddress ReadyTimePickup" \
             " CloseTimePickup PickupType WrongDayLateCount RightDayLateCount FedExID Longitude Latitude"

    fmt = ['%.8d', '%.2d', '%.2d', '%.4d', '%.6d', '%.4d', '%.4d', '%.4d', '%.4d', '%.4d', '%.4d', '%.4d',
           '%.4d', '%.18f', '%.18f']

    np.savetxt('fedex_pc_cleaned_no_0.data', array, fmt=fmt, header=header)
    print("File written, hope it's fine !")


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

    # TODO: retrieve geocode for the next batch 0-2000 => poor results...
    # Location to ID lookup and vice-versa
    with open("DATA/postal_codes_fedex", 'rb') as f:
        id_to_pc = pickle.load(f)

    pc_to_id = {j: i for i, j in enumerate(id_to_pc)}

    # pc to coordinates using API

    increase_pc_list = sorted(pc_to_id.keys())
    print("Total postal code to retrieve ", len(increase_pc_list))
    pc_l1 = increase_pc_list[2001:4000]  # day 2 of stealing google

    pc_to_coords = {}  # store results with a dict {postalcode : {'lng': ?? , 'lat' : ??}}
    count_err = 0
    for postal_code in pc_l1:

        params = { "address" : "Singapore " + str(postal_code),
                   "sensor" : "false",
                   "region" : "SG",
                    'key' : API_key
                  }
        try:
            results = requests.get(
                url,
                params=params
            ).json()


            # append coords at the postal code
            pc_to_coords[postal_code] = {
                'lng':results['results'][0]['geometry']['location']['lng'],
                'lat':results['results'][0]['geometry']['location']['lat']
            }
        except:
            pc_to_coords[postal_code] = 'error'
            print("error retrieving")
            count_err += 1

    # save pc_to_coords
    with open('/Users/Louis/PycharmProjects/policy_approximation/DATA/pc_to_coordinates/pc_to_coord_2001_4000', "wb") as f:
        pickle.dump(pc_to_coords, f)

    print("Number of errors while retrieving coordinates: ", count_err)

    """
    # TODO: regroup this in a function in order to plot easily on the same map different color point
    # --------------------- VISUALIZATION -----------------------------
    # input: postal codes, color (to differentiate addresses nature)
    with open('/Users/Louis/PycharmProjects/policy_approximation/DATA/pc_to_coordinates/pc_to_coord_0_2000', "rb") as f:
        pc_to_coord = pickle.load(f)

    lat = [pc_to_coord[i]['lat'] for i in pc_to_coord]
    lng = [pc_to_coord[i]['lng'] for i in pc_to_coord]

    viz(lat, lng)
    """
