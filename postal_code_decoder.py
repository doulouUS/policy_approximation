import geocoder
import pandas as pd
import pickle


def ID_to_address(id):
    addresses_path = "/Users/Louis/PycharmProjects/MEng_Research/foo-Environment_2/gym_foo/envs/addresses.fedex"
    with open(addresses_path, "rb") as f:
        addresses = pickle.load(f)

    return addresses[id]

raw_data_path = "/Users/Louis/PycharmProjects/MEng_Research/foo-Environment_2/dynamics/demand_models/fedex.data"
DATA_FRAME = pd.read_csv(raw_data_path, header=0, delim_whitespace=True)
count_true = 0

for ID in range(0, 300):
    add_str = ID_to_address(ID)  # string address, address
    # print(add_str)

    postal_code = geocoder.google(add_str[0]+", Singapore")
    # print("geocoder retrieved code ", postal_code.postal)
    # print("data code ", DATA_FRAME["PostalCode"][ID])

    if DATA_FRAME.loc[DATA_FRAME["Address"] == ID]["PostalCode"][0] == postal_code.postal :
        count_true += 1


print("Google ability to retrieve postal code from string address ", count_true/300)