import numpy as np
import random
import datetime
from datetime import tzinfo
import math
from math import radians, sin, cos, asin, sqrt
import pandas as pd
import numpy.polynomial.polynomial as poly
import sys
sys.path.insert(0, '/Users/Louis/PycharmProjects/policy_approximation')
from models_and_data_reader.read_data_fedex import basic_settings as base_data
from models_and_data_reader.read_data_fedex import initialize_map, add_line_on_map, add_point_on_map, show_map
from models_and_data_reader.svm_classification import train_svm_model

from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as md

# API call Google
import requests

PATH_TO_DATA = "/Users/Louis/PycharmProjects/policy_approximation/DATA/fedex_pc_cleaned_no_0.data"
# Loading file in memory to speed up reading
PC_COORD = pd.read_csv("/Users/Louis/PycharmProjects/policy_approximation/DATA/"
                       "pc_to_coordinates/PostalCodeCoordinates.csv",
                       sep=",")
STATS = pd.read_csv("/Users/Louis/PycharmProjects/policy_approximation/simulator/traveltime_stats.txt", sep=" ")
# --------
class GMT8(tzinfo):
    """
    Implement the tzinfo object specific to Singapore time zone
    """
    def utcoffset(self, dt):
        return datetime.timedelta(hours=8, minutes=0)

    def tzname(self, dt):
        return "GMT +8"

    def dst(self, dt):
        return datetime.timedelta(0)

gmt8 = GMT8()

def haversine(lat1, lon1, lat2, lon2):
    """
    Return the approximate distance between the two coordinates
    :param lat1: float
    :param lon1: float
    :param lat2: float
    :param lon2: float
    :return: distance in km
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    km = 6367 * c
    return km


def mean_time(list_time):
    return sum(list_time) / len(list_time)


def nearest(items, pivot):
    """
    Generic function to return the element in items that is the closest to pivot
    :param items: list
    :param pivot: object that support comparison, subtraction, abs
    :return: nearest object to pivot in items that support comparison, subtraction, abs,
    """
    return min(items, key=lambda x: abs(x - pivot))


def get_coordinates(postal_code):
    """
    Return the coordinates corresponding to the nearest postal code present in
    /Users/Louis/PycharmProjects/policy_approximation/DATA/pc_to_coordinates/PostalCodeCoordinates.csv

    :param postal_code: int
    :return: tuple (Lat, Long) so that it is immutable and can be used as a python dict
    """
    # TODO IMPROVE: ideally we want the exact coordinates of postal_code not the ones of the closest...
    # TODO IMPROVE: ...postal code !!
    # we pre loaded PC_COORD to speed up computations
    name = PC_COORD.ix[(PC_COORD['Postal Code']-postal_code).abs().argsort()[0]]
    return (name.Lat, name.Long)


def filter_lower_datetime(time, list_time):
    """
    Return datetime elements from list_time lower than time
    :param time: datetime.datetime object
    :param list_time: list of datetime.datetime objects
    :return: list of datetime.datetime objects
    """
    return [t for t in list_time if t <= time]


def get_travel_time(pc1, pc2, time, rand=False, mode="ggAPI"):
    # TODO: implement the random part if needed
    # TODO: possible to replace this by API calls to Google Maps in real time
    """
    Travel time retriever, used in the simulator
    :param pc1: int, postal code of departure
    :param pc2: int, postal code of arrival
    :param time: datetime.datetime object, time at departure
    :param rand: bool, if True, travel time is sampled from a normal distribution
    :param mode: str, ggAPI or simple.
    :return: tuple, datetime.timdelta object, time needed to travel from pc1 to pc2 leaving at time + distance
    """
    if mode == "simple":
        # Retrieve distance between pc1 and pc2
        # #pc -> coord (read data)
        closest1 = get_coordinates(pc1)
        closest2 = get_coordinates(pc2)

        # print("closest to", pc1, " is ", closest1)
        # print("closest to ", pc2, " is ", closest2)

        # #coords -> distance (haversine)
        dist = haversine(closest1[0], closest1[1], closest2[0], closest2[1])

        # Load the appropriate statistics
        times = [datetime.datetime.strptime(str(t), '%H%M').replace(year=1970,
                                                                    month=1,
                                                                    day=1,
                                                                    tzinfo=gmt8)
                 for t in list(STATS["DAYTIME"][1:])]  # avoid the 0-line
        idx_nearest_time = times.index(nearest(times, time))
        # print(idx_nearest_time)
        poly_coeff = list(STATS[["C","B","A"]].iloc[idx_nearest_time+1])  # correction of +1 to avoid the 0-line...
        # print("COEF", poly_coeff)

        # apply the model -> retrieve mean and std (optional) ADDED 10 mn of service, this model is way too optimistic !
        return datetime.timedelta(seconds=poly.polyval(dist, poly_coeff)+10*60), dist
# --------
    elif mode == "ggAPI":
        # coords
        # #pc -> coord (read data)
        closest1 = get_coordinates(pc1)
        closest2 = get_coordinates(pc2)

        # retrieve UTC time in seconds since 1970-1-1
        cur_time = datetime.datetime.now(tz=gmt8)
        time_traffic = time.replace(year=cur_time.year, month=cur_time.month, day=cur_time.day+1, tzinfo=gmt8)
        # +1 to avoid past call to Google API

        # api call
        url = "https://maps.googleapis.com/maps/api/directions/json?"
        params = {"origin": str(closest1[0])+','+str(closest1[1]),
                  "destination":str(closest2[0])+','+str(closest2[1]),
                  "key": "AIzaSyApMtNEiMXlpRju4TR8my3lK_0tG-VafPU",
                  "units":"metrics",
                  "departure_time":str(int(time_traffic.timestamp() // 1)),
                  "region":"sg",
                  "traffic_model":"best_guess"
                  }
        api_call = requests.get(
            url,
            params=params
        ).json()

        if api_call["status"] == 'INVALID_REQUEST':
            raise ValueError('Invalid Google Maps API request: '+api_call["error_message"])

        # duration and distance retrieval (seconds and meters)
        duration = api_call["routes"][0]["legs"][0]["duration"]["value"]
        dist = api_call["routes"][0]["legs"][0]["distance"]["value"]

        return datetime.timedelta(seconds=duration + 3*60), dist



    else:
        raise ValueError('Incorrect mode :'+mode+'. Choose between "simple" and "ggAPI".')


class State:

    def __init__(self, initial_location, initial_time, D_0, P_0):
        """

        :param initial_location: int, postalcode
        :param initial_time: datetime.datetime object
        :param D_0: dict, postal codes:[pickupHour1, pickupHour2, ...]
        :param P_0: dict, postal codes:nb_of_remaining_deliv
        """
        # Index
        self.k = 0

        # current time
        self.t_k = initial_time

        # current location
        self.c_k = initial_location

        # remaining deliveries - dict postalcode:[Lat, Long]
        self.D_k = {pc: [D_0[pc], get_coordinates(pc)] for pc in D_0.keys()}

        # remaining pickups - same
        self.P_k = {pc:[P_0[pc], get_coordinates(pc)] for pc in P_0.keys()}

    def __repr__(self):
        output_string =  "Step "+str(self.k)+"\n" \
                        "Current time       t_"+str(self.k)+": "+str(self.t_k)+"\n"\
                        "Current location   c_" + str(self.k) + ": " + str(self.c_k) + "\n"\
                        "Remaining deliv    D_" + str(self.k) + ": " + str(self.D_k.keys()) + "\n"\
                        "Remaining pickup   P_" + str(self.k) + ": " + str(self.P_k) + "\n"
        return output_string


class PerfTracker:
    """
    To keep track of values and any data generated by the simulation
    """
    def __init__(self, simulator, mode_scenario):
        self.simulator = simulator

        # time t_k
        self.time_t_k = [self.simulator.state.t_k]

        # nb of total known jobs at cur step
        nb_deliv = [nb[0] for nb in self.simulator.state.D_k.values()]
        self.total_nb_known_jobs_k = [
            len(self.simulator.served_deliv_pc) + \
            sum([len(length) for length in self.simulator.served_pickup_pc.values()]) + \
            sum([len(length) for length in self.simulator.state.P_k.values()]) + \
            sum(nb_deliv)
        ]

        # total nb of served jobs at each step - deliv and pickups
        self.nb_served_jobs_k = [len(self.simulator.served_deliv_pc) + len(self.simulator.served_pickup_pc.keys())]

        # value of the objective function at each step
        self.obj_func_k = [len(self.simulator.served_deliv_pc)*0.5 + len(self.simulator.served_pickup_pc.keys())*1]

        # total distance since start
        self.total_distance = [0]

        # total waiting time since start
        self.total_waiting_periods = 0

        # average time between two deliveries/pickups OR average deliv/pickups per hour
        self.average_deliv_h = []
        self.average_pickup_h = []

        # historical events: if mode="realistic"
        # list of times
        self.mode_scenario = mode_scenario
        if self.mode_scenario == "realistic":
            self.hist_t_k = [datetime.datetime.strptime(str(int(row["StopStartTime"])), '%H%M').replace(
                            year=1970,
                            month=1,
                            day=1,
                            tzinfo=gmt8)
                for idx, row in self.simulator.data_scenario.iterrows()]

            print("SET PC", set(simulator.data_scenario["PostalCode"]))

            # nb of jobs done
            self.done_hist_jobs = range(1, len(self.hist_t_k)+1)


    def __repr__(self):
        output_string = "-----TRACKER SUMMARY------\n" \
                        "\n" \
                        "Successive times                   : "+str(self.time_t_k)+"\n" \
                        "Total nb of known jobs             : "+str(self.total_nb_known_jobs_k)+"\n" \
                        "Total nb of served jobs            : "+str(self.nb_served_jobs_k)+"\n" \
                        "Total distance for each step       : "+str(self.total_distance)+"\n" \
                        "Total waiting periods              : "+str(self.total_waiting_periods)+"\n"

        return output_string

    def update_tracker(self, simulator, distance, decision, delta_time=0):
        # time t_k
        self.time_t_k.append(simulator.state.t_k)

        # nb of total known jobs at cur step
        nb_deliv = [nb[0] for nb in self.simulator.state.D_k.values()]
        self.total_nb_known_jobs_k.append(
            len(self.simulator.served_deliv_pc) + \
            sum([len(length) for length in self.simulator.served_pickup_pc.values()]) + \
            sum([len(length) for length in self.simulator.state.P_k.values()]) + \
            sum(nb_deliv)
        )

        # nb of served jobs at each step - deliv and pickups
        self.nb_served_jobs_k.append(len(simulator.served_deliv_pc) +
                                     sum([len(length) for length in self.simulator.served_pickup_pc.values()])
                                     )

        # value of the objective function at each step
        self.obj_func_k.append(len(simulator.served_deliv_pc)*1 +
                               sum([len(length) for length in self.simulator.served_pickup_pc.values()])*0.5)

        # total distance since start
        self.total_distance.append(self.total_distance[-1] + distance)

        # total waiting time since start
        if decision == 0:
            self.total_waiting_periods += 1

        # average time between two deliveries/pickups OR average deliv/pickups per hour
        elapsed_time = simulator.state.t_k - simulator.departure_time
        self.average_deliv_h.append(len(simulator.served_deliv_pc) / (elapsed_time.total_seconds()/60))
        self.average_pickup_h.append(sum([len(length) for length in self.simulator.served_pickup_pc.values()])
                                     / (elapsed_time.total_seconds()/60))

        # average time between two deliveries

    def plot_summary(self, suptitle="Mention Policy Here"):
        # Map
        # show_map(plot=plot)

        # Curves to visualize results
        dates = md.datestr2num([t.strftime("%H:%M") for t in self.time_t_k])
        if self.mode_scenario == "realistic":
            dates_hist = md.datestr2num([t.strftime("%H:%M") for t in self.hist_t_k])

        # HISTOGRAM
        figure = plt.figure()
        figure.suptitle(suptitle)
        axes = [figure.add_subplot(2, 2, i) for i in range(1, 5)]

        # Ticker and x-axis formatter
        hour_tick = matplotlib.dates.HourLocator()
        hour_formatter = matplotlib.dates.DateFormatter("%H")
        # First nb of known jobs + nb of served jobs
        axes[0].plot_date(dates, self.total_nb_known_jobs_k, c='blue')  # , fillstyle="full", markeredgewidth=0.0)
        axes[0].plot_date(dates, self.nb_served_jobs_k, c="black")  # , fillstyle="full", markeredgewidth=0.0)
        # historical data
        if self.mode_scenario == "realistic":
            axes[0].plot_date(dates_hist, self.done_hist_jobs, c='red')  # , fillstyle="full", markeredgewidth=0.0)
            # axes[0].set_xticklabels([t.time() for t in self.time_t_k])

        axes[0].xaxis.set_major_locator(hour_tick)
        axes[0].xaxis.set_major_formatter(hour_formatter)
        axes[0].set_title("Number of jobs done and nb of total available jobs")
        # axes[0].set_fillstyle('bottom')

        axes[1].plot_date(x=dates, y=self.total_distance, linestyle='solid',markeredgewidth=0.0)
        axes[1].xaxis.set_major_locator(hour_tick)
        axes[1].xaxis.set_major_formatter(hour_formatter)
        axes[1].set_title("total distance vs time")

        axes[2].plot_date(x=dates, y=self.obj_func_k, linestyle='solid',markeredgewidth=0.0)
        axes[2].xaxis.set_major_locator(hour_tick)
        axes[2].xaxis.set_major_formatter(hour_formatter)
        axes[2].set_title("objective function vs time")

        axes[3].plot_date(x=dates[1:], y=self.average_deliv_h, linestyle='solid')
        axes[3].xaxis.set_major_locator(hour_tick)
        axes[3].xaxis.set_major_formatter(hour_formatter)
        axes[3].set_title("average deliv rate (per hour)")

        return figure, axes


class Simulator:
    """
    Simulator
    for a given set of locations
    truck(s) ID
    delivery jobs (with possibilty of arranging them at will)
    number of pickups that will appear (modifying at will: reflecting a distribution)

    next state generator:
            inputs: action, state
            outputs: next state
    """
    def __init__(self, scenario_mode="random", policy="computer_driver_heuristic", nb_truck=15,
                 nb_deliv_jobs=5, nb_pickup_jobs=5, seed=10):

        self.policy = policy  # str
        self.nb_deliv_jobs = nb_deliv_jobs
        self.nb_pickup_jobs = nb_pickup_jobs

        self.departure_location = 498746
        self.departure_coord = [1.313832, 103.8262954]

        # BASE SETTINGS
        if scenario_mode == "random":
            self.data, self.data_indices, _, self.sorted_truck, self.idx_to_pc, self.pc_to_idx \
                = base_data(nb_truck=nb_truck)

            # SCENARIO GENERATOR
            # pickup jobs
            # pickup location candidates
            data = self.data[self.data["ReadyTimePickup"] != 0]  # work on pickup data only
            pickup_candidates = [int(pc) for pc in self.idx_to_pc if len(data[data["PostalCode"] == pc]) > 0]
            # random.seed(seed)

            # selection of pickups
            self.pc_pickup_jobs = random.sample(pickup_candidates, nb_pickup_jobs)

            # delivery jobs among remaining locations (arrangement)
            # TODO ideally here sample the KDE model of the day in the region
            # selection of deliveries # - set(self.pc_pickup_jobs) not implemented: possible overlap btw delivery
            #                                                               and pickup
            pc_deliv_jobs = random.sample(set(self.idx_to_pc), nb_deliv_jobs)
            self.delivery_loc_to_nb = {loc: random.randint(1, 5) for loc in pc_deliv_jobs}

        elif scenario_mode == "realistic":
            self.data, _, self.data_scenario, _, self.idx_to_pc, self.pc_to_idx \
                = base_data(nb_truck=nb_truck)

            # SCENARIO GENERATOR
            # deliveries locations: set as once visited it is served
            # TODO change all self.pc_deliv_jobs into delivery_loc_to_nb in "realistic" mode then "random"

            # self.pc_deliv_jobs = \
            #    list(set(self.data_scenario[self.data_scenario["ReadyTimePickup"] == 0]["PostalCode"].values))
            self.delivery_loc_to_nb = dict()
            for idx, row in self.data_scenario[self.data_scenario["ReadyTimePickup"] == 0].iterrows():
                # first appearance of loca
                if row["PostalCode"] not in self.delivery_loc_to_nb.keys():
                    self.delivery_loc_to_nb[row["PostalCode"]] = 1
                else:
                    self.delivery_loc_to_nb[row["PostalCode"]] += 1

            # TODO END
            # pickup locations: not a set as there can be repeated location with different time
            self.pc_pickup_jobs = \
                list(self.data_scenario[self.data_scenario["ReadyTimePickup"] != 0]["PostalCode"].values)
        else:
            raise ValueError("In Simulator object creation, please choose a valid scenario mode. You input "
                             +str(scenario_mode))

        # dict containing the jobs
        self.all_jobs_pc_to_coord = dict()
        self.all_jobs_pc_to_coord["pickups"] = {pc:get_coordinates(pc) for pc in self.pc_pickup_jobs}
        self.all_jobs_pc_to_coord["deliveries"] = {pc:get_coordinates(pc) for pc in self.delivery_loc_to_nb.keys()}
        # print("dict of all jobs ", self.all_jobs_pc_to_coord)

        self.all_jobs_coord_to_pc = dict()
        self.all_jobs_coord_to_pc["pickups"] = {coord:pc for pc, coord in self.all_jobs_pc_to_coord["pickups"].items()}
        self.all_jobs_coord_to_pc["deliveries"] = {coord: pc for pc, coord in self.all_jobs_pc_to_coord["deliveries"].items()}

        self.pc_to_coord = dict()
        for type_job in ["pickups", "deliveries"]:
            for pc, coord in self.all_jobs_pc_to_coord[type_job].items():
                self.pc_to_coord[pc] = coord
        self.pc_to_coord[self.departure_location] = self.departure_coord

        # TODO Pb ! we must re-create realistic scenarios. This part is just a headstart
        # when
        if scenario_mode == "random":

            self.pickup_loc_to_time = defaultdict(list)

            pickup_arrival_time_mean, pickup_arrival_time_var = self.get_ready_time()
            for location in self.pc_pickup_jobs:
                for rep in range(0, random.randint(1, 4)):
                    self.pickup_loc_to_time[location].append(
                        datetime.datetime(
                            year=1970,
                            month=1,
                            day=1,
                            hour=0,
                            minute=0,
                            second=0,
                            microsecond=0,
                            tzinfo=gmt8)
                            +
                            datetime.timedelta(seconds=math.floor(random.gauss(
                                pickup_arrival_time_mean[int(location)],
                                pickup_arrival_time_var[int(location)]))
                            )
                    )
            print("PICKUP_LOC_TO_TIME  : ", self.pickup_loc_to_time)
        elif scenario_mode == "realistic":

            self.pickup_loc_to_time = defaultdict(list)

            for idx, row in self.data_scenario[self.data_scenario["ReadyTimePickup"] != 0].iterrows():
                self.pickup_loc_to_time[row["PostalCode"]].append(
                    datetime.datetime.strptime(str(int(row["StopStartTime"])), '%H%M').replace(
                        year=1970,
                        month=1,
                        day=1,
                        tzinfo=gmt8)
                )
        else:
            raise ValueError("In Simulator object creation, please choose a valid scenario mode. You input "
                             +str(scenario_mode))

        self.pickup_time_to_loc = dict()
        for loc, time in self.pickup_loc_to_time.items():
            for i in time:
                self.pickup_time_to_loc[i] = loc

        # Keep history of states with a list of State object
        self.served_pickup_pc = defaultdict(list)  # {postalcode: [timeReady1, timeReady2, ...]}
        self.served_deliv_pc = []

        # State
        self.departure_time = datetime.datetime(year=1970,
                                                month=1,
                                                day=1,
                                                hour=9,
                                                minute=40,
                                                tzinfo=gmt8)

        # Initialize state by filtering unknown deliveries at departure time
        # TODO Solve this question of the departure location ! For now, 1 chosen among the deliveries
        self.state = State(initial_location=self.departure_location,  # self.departure_location,
                           initial_time=self.departure_time,
                           P_0={self.pickup_time_to_loc[t]:[self.pickup_loc_to_time[self.pickup_time_to_loc[t]]]
                                for t in filter_lower_datetime(self.departure_time,
                                                               [item for sublist in self.pickup_loc_to_time.values()
                                                                for item in sublist])
                                },
                           D_0=self.delivery_loc_to_nb
                           )
        # Perf tracker
        self.tracker = PerfTracker(self, mode_scenario=scenario_mode)

        # Policies
        # Train and retrieve the model
        if self.policy == "computer_driver_heuristic":
            self.heuristic_model = train_svm_model(nb_truck=nb_truck,method="classification", percentage=1)

    def get_ready_time(self):
        """
        Generate from historical data, mean and var of pickup appearance for each location assigned to be a pickup
        :return: tuple (2,): dict of mean and dict of var for each postal code in self.pc_pickup_jobs
        """

        # historical data
        data_pickup = self.data[self.data["ReadyTimePickup"] != 0]  # work on pickup data only
        pickup_arrival_times = {
            location: [
                (datetime.datetime.strptime(str(int(row["ReadyTimePickup"])), '%H%M').replace(year=1970,
                                                                                              month=1,
                                                                                              day=1,
                                                                                              tzinfo=gmt8)
                 - datetime.datetime.now().replace(
                    year=1970,
                    month=1,
                    day=1,
                    hour=0,
                    minute=0,
                    second=0,
                    microsecond=0,
                    tzinfo=gmt8)
                 ).total_seconds()
                for _, row in data_pickup[data_pickup["PostalCode"] == location].iterrows()
                ]
            for location in self.pc_pickup_jobs
            }

        # mean
        pickup_arrival_time_mean = {
            location: mean_time(pickup_arrival_times[location])
            for location in self.pc_pickup_jobs
        }
        # variance
        pickup_arrival_time_var = {
            location: np.std(pickup_arrival_times[location])/2
            for location in self.pc_pickup_jobs
        }

        return pickup_arrival_time_mean, pickup_arrival_time_var

    def update_remaining_jobs(self, decision):
        """
        Update the remaining jobs based on the visited place decision.
        If decision is a deliveries, it simply removes it from self.state.D_k and self.state.coord_D_k.
        If decision is a pickup, it removes it from self.state.P_k and self.state.coord_Pk, and
        update potential new pickups, from the scenario.
        :param decision: int, postal code
        :return: nothing
        """
        # TODO pb if several times the same postal code in the remaining jobs
        # TODO is it good to keep two lists of the locations, postal codes and dataframes containing coords
        if decision in list(self.state.D_k.keys()):
            if self.state.D_k[decision][0] > 1:
                self.state.D_k[decision][0] -= 1
            else:
                del self.state.D_k[decision]  # all jobs at decision have been served
            self.served_deliv_pc.append(decision)  # record the action

        # remaining pickups
        elif decision in list(self.state.P_k.keys()):
            # remove the minimum time corresponding to the decision
            time_ready = min(self.state.P_k[decision][0])
            if len(self.state.P_k[decision][0]) > 1:
                self.state.P_k[decision][0] = [t for t in self.state.P_k[decision][0]
                                               if t != min(self.state.P_k[decision][0])]
            else:
                del self.state.P_k[decision]
            self.served_pickup_pc[decision].append(time_ready)

        elif decision == 0:  # wait case, no removing, so we go the update part directly
            pass

        else:
            raise ValueError('Error in simulator\'s method update_remaining_jobs: ' \
                             'decision is not in self.state.D_k or self.state.P_k keys')

        # update by looking at the pre-loaded scenario in self.pickup_loc_to_time, not already served
        new_jobs = {t : self.pickup_time_to_loc[t]
                    for t in filter_lower_datetime(self.state.t_k, self.pickup_time_to_loc.keys())
                    if [self.pickup_time_to_loc[t], t] not in
                    [
                        [place, t] for place, times in self.served_pickup_pc.items() for t in times
                    ]
                    +
                    [
                        [place, t] for place, times in self.state.P_k.items() for t in times[0]
                    ]
                    }
        # new_jobs = [job for job, t in self.pickup_loc_to_time.items() if min(t) < self.state.t_k
        #            and (len(t) > self.served_pickup_pc.count(job) or job not in self.served_pickup_pc)]

        for new_time, new_job in new_jobs.items():
            new_job_coord = get_coordinates(new_job)
            if new_job not in self.state.P_k.keys():
                self.state.P_k[new_job] = [[new_time], new_job_coord]
            else:
                self.state.P_k[new_job][0].append(new_time)  # no need to add coordinates

    def display_scenario(self):
        print("Scenario:")
        print("     Pickups")

        for loc, time in self.pickup_loc_to_time.items():
            print("         ", loc, time)

        print("____________")
        print("     Deliveries")

        for loc in self.delivery_loc_to_nb.keys():
            print("         ", loc, self.delivery_loc_to_nb[loc])

    # --------
    #   Policies

    def nearest_neigbor(self, pc):
        """
        Return the closest place from pc among self.delivery_loc_to_nb.keys() and self.pc_pickup_jobs
        :param pc: int, postal code
        :return: int, postal code. 0 if no more jobs
        """
        coord = get_coordinates(pc)
        # deliveries
        pdist_deliv = {haversine(coord[0], coord[1], pcoord[1][0], pcoord[1][1]):pc for pc, pcoord in self.state.D_k.items()}
        pdist_list_deliv = list(pdist_deliv.keys())
        if len(pdist_list_deliv) > 0:
            val_deliv_min = min(pdist_list_deliv)
        else:
            val_deliv_min = 1e6  # great value to be discarded when comparing with val_pickup_min
        # pickups
        pdist_pickup = {haversine(coord[0], coord[1], pcoord[-1][0], pcoord[-1][1]):pc for pc, pcoord in self.state.P_k.items()}
        pdist_list_pickup = list(pdist_pickup.keys())

        if len(pdist_list_pickup) > 0:
            val_pickup_min = min(pdist_list_pickup)
        else:
            val_pickup_min = 1e6  # great value to be discarded when comparing with val_pickup_min

        if val_deliv_min == val_pickup_min and val_deliv_min == 1e6:
            print("All jobs completed: go to wait or stop if it's 12pm")
            return 0

        if val_deliv_min < val_pickup_min:
            return pdist_deliv[val_deliv_min]

        elif val_deliv_min >= val_pickup_min:
            return pdist_pickup[val_pickup_min]
        else:
            raise valueError('Impossible comparison between val_deliv_min and val_pickup_min ')

    def computer_driver_heuristic(self, pc):
        """
        Return the decision (postal code) of our trained heuristic among self.state.D_k.keys() or self.state.P_k.keys()

        /!\ SPECIAL CASE WHEN AT THE DEPOT : pc == 498746
        :param: pc, int: postal code of the current location
        :return: int, postal code. 0 if no more jobs
        """
        if pc == self.departure_location:
            return self.nearest_neigbor(pc)
        else:
            # encode state: State -> Generalized One hot vector
            # print(len(self.idx_to_pc)+1)
            encoded_vector = np.zeros(len(self.idx_to_pc)+1)

            # indices of locations FOR ENCODING
            pickup_jobs_idx = [self.pc_to_idx[p]+1 for p in list(self.state.P_k.keys())]  # +1 is to make room for the time dim
            deliv_jobs_idx = [self.pc_to_idx[p]+1 for p in list(self.state.D_k.keys())]

            # indices of locations FOR PC READING
            pickup_jobs_idx_read = [self.pc_to_idx[p] for p in list(self.state.P_k.keys())]
            deliv_jobs_idx_read = [self.pc_to_idx[p] for p in list(self.state.D_k.keys())]
            tasks = set(pickup_jobs_idx_read + deliv_jobs_idx_read)

            if len(tasks) > 0:
                # set appropriate values at the index corresponding to the location
                encoded_vector[pickup_jobs_idx] = -0.5
                encoded_vector[deliv_jobs_idx] = 0.5
                encoded_vector[self.pc_to_idx[pc]+1] = 1

                # # current time encoded as nb of seconds between 12pm and now/nb seconds between 12pm and 12am
                total_nb_seconds = datetime.timedelta(hours=12, minutes=0)
                cur_time = self.state.t_k.time()
                cur_time = datetime.timedelta(hours=cur_time.hour, minutes=cur_time.minute)  # nb of seconds from 12am
                # # TODO this can further be noramlized as most values will be >0 (>6am)
                cur_time = 2 * cur_time.seconds / total_nb_seconds.seconds - 1  # normalized time in [-1,1]
                encoded_vector[0] = cur_time

                # predict decision
                pred = self.heuristic_model.predict_proba(encoded_vector.reshape(1,-1))

                # take the most probable location among the remaining jobs
                # # set proba to 0 if location not among remaining jobs
                # print("##############")
                # print("shape of pred ", pred.shape)
                # print("Number of locations considered  : ", len(self.idx_to_pc))
                print("Possible indices to choose from : ", tasks)
                pred[0, list(set(range(0, len(self.idx_to_pc))) - set(pickup_jobs_idx_read + deliv_jobs_idx_read))] = 0

                idx_opt = np.argsort(pred[0,:])[-1]  # most probable location (by its index) among remaining jobs
                print("Index chosen                     : ", idx_opt )
                return self.idx_to_pc[idx_opt]

            elif len(tasks) == 0:
                return 0

            else:
                raise ValueError('Problem with tasks, which has negative length...')

    def random_policy(self, pc):
        """
        Implement a random policy for comparison
        :param pc: int , postal code
        :return: int, postal code
        """
        if list(self.state.P_k.keys()) or list(self.state.D_k):
            return random.choice(list(self.state.P_k.keys())+list(self.state.D_k))
        elif not(list(self.state.P_k.keys()) and list(self.state.D_k)):
            return 0
        else:
            raise ValueError('Check jobs remaining to be done.')


    def next_state(self, decision):
        # TODO Static?
        """
        Return the next state after taking decision, with current state
        :param decision: int, postal code (among self.pc_deliv_jobs and self.pc_pickup_jobs !!)
        :return: tuple, decision int (postalcode) and delta_time int (seconds)
        """
        if decision != 0:
            # Current index
            self.state.k += 1
            # current time
            delta_time, distance = get_travel_time(self.state.c_k, decision, self.state.t_k, mode="ggAPI")
            self.state.t_k += delta_time

            # current location
            self.state.c_k = decision

            # remaining deliveries
            self.update_remaining_jobs(decision=decision)

            return delta_time, distance

        elif decision == 0:  # wait
            # Current index: no modification

            # current time: wait
            self.state.t_k += datetime.timedelta(seconds=2*60)

            # current location: no modification

            # remaining deliveries
            self.update_remaining_jobs(decision=decision)

            return 2*60, 0

    def run_simulator(self, noon_deadline, plot=initialize_map()):

        sim.display_scenario()

        print("start time ", sim.state.t_k)
        while self.state.t_k < noon_deadline:
            # print state
            print(self.state)

            print("-----------------------------")
            # print decision
            print("** DECISION ** ")
            if self.policy == "nearest_neighbor":
                decision = self.nearest_neigbor(sim.state.c_k)
            elif self.policy == "computer_driver_heuristic":
                decision = self.computer_driver_heuristic(sim.state.c_k)
            elif self.policy == "random":
                decision = self.random_policy(sim.state.c_k)
            else:
                raise ValueError('In Simulator\'s run_simulator, wrong policy name: %s' % policy)
            print("Decision where to go next  : ", decision)

            # plotting
            if decision != 0:
                add_point_on_map(lat=[sim.pc_to_coord[sim.state.c_k][0]],
                                 lng=[sim.pc_to_coord[sim.state.c_k][1]],
                                 plot=plot,
                                 label="step " + str(sim.state.k))
            # compute new state
            print("** NEW STATE ** ")
            delt_time, dist = self.next_state(decision=decision)

            # update tracker
            self.tracker.update_tracker(self, distance=dist, decision=decision, delta_time=delt_time)

        print("END OF OPERATION")
        return plot, self.tracker


if __name__ == '__main__':
        # get_travel_time(119114, 120100, time=datetime.datetime.now(),mode="ggAPI")
        policy = "nearest_neighbor"  # "nearest_neighbor" or "computer_driver_heuristic" or "random"
        scenario_mode = "random"  # "realistic" or "random"
        # Simulator
        sim = Simulator(scenario_mode=scenario_mode, policy=policy, nb_truck=5)
        noon_deadline = datetime.datetime(year=1970,
                                          month=1,
                                          day=1,
                                          hour=12,
                                          minute=0,
                                          second=0,
                                          microsecond=0,
                                          tzinfo=gmt8)

        # decision = sim.computer_driver_heuristic(list(sim.state.D_k.keys())[0])
        # print("Decision made by the computer :" , decision)

        # Run
        plot, tracker = sim.run_simulator(noon_deadline=noon_deadline)  # or "computer_driver_heuristic"

        # Tracker info
        print(tracker)
        print(" ")
        print("List of visited deliveries, chronologically   : ", sim.served_deliv_pc)
        print("List of visited pickups, chronologically      : ", sim.served_pickup_pc)

        # Plot tracker summary
        fig, ax = tracker.plot_summary(suptitle=policy)

        plt.show()

        # TODO improve coordinates of our postal codes

        # TODO evaluate performance of our SVM when filtering the results to allow only known jobs. ok
        # TODO Draw on map the order number on top of the lines

