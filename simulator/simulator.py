import numpy as np
import random
import datetime
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

PATH_TO_DATA = "/Users/Louis/PycharmProjects/policy_approximation/DATA/fedex_pc_cleaned_no_0.data"
# Loading file in memory to speed up reading
PC_COORD = pd.read_csv("/Users/Louis/PycharmProjects/policy_approximation/DATA/"
                       "pc_to_coordinates/PostalCodeCoordinates.csv",
                       sep=",")
STATS = pd.read_csv("/Users/Louis/PycharmProjects/policy_approximation/simulator/traveltime_stats.txt", sep=" ")
# --------


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

def get_travel_time(pc1, pc2, time, rand=False):
    # TODO: implement the random part if needed
    # TODO: possible to replace this by API calls to Google Maps in real time (so day-long code running)
    """
    Travel time retriever, used in the simulator
    :param pc1: int, postal code of departure
    :param pc2: int, postal code of arrival
    :param time: datetime.datetime object, time at departure
    :param rand: bool, if True, travel time is sampled from a normal distribution
    :return: tuple, datetime.timdelta object, time needed to travel from pc1 to pc2 leaving at time + distance
    """
    # Retrieve distance between pc1 and pc2
    # #pc -> coord (read data)
    closest1 = get_coordinates(pc1)
    closest2 = get_coordinates(pc2)

    # print("closest to", pc1, " is ", closest1)
    # print("closest to ", pc2, " is ", closest2)

    # #coords -> distance (haversine)
    dist = haversine(closest1[0], closest1[1], closest2[0], closest2[1])

    # Load the appropriate statistics
    times = [datetime.datetime.strptime(str(time), '%H%M').replace(year=1970,
                                                                   month=1,
                                                                   day=1,
                                                                   tzinfo=None)
             for time in list(STATS["DAYTIME"][1:])]  # avoid the 0-line
    idx_nearest_time = times.index(nearest(times, time))
    # print(idx_nearest_time)
    poly_coeff = list(STATS[["C","B","A"]].iloc[idx_nearest_time+1])  # correction of +1 to avoid the 0-line...
    # print("COEF", poly_coeff)

    # apply the model -> retrieve mean and std (optional) ADDED 10 mn of service, this model is way too optimistic !
    return datetime.timedelta(seconds=poly.polyval(dist, poly_coeff)+10*60), dist
# --------


class State:

    def __init__(self, initial_location, initial_time, D_0, P_0):
        """

        :param initial_location: int, postalcode
        :param initial_time: datetime.datetime object
        :param D_0: list, list of postal codes
        :param P_0: list, list of postal codes
        """
        # Index
        self.k = 0

        # current time
        self.t_k = initial_time

        # current location
        self.c_k = initial_location

        # remaining deliveries - dict postalcode:[Lat, Long]
        self.D_k = {pc:get_coordinates(pc) for pc in D_0}

        # remaining pickups - same
        self.P_k = {pc:get_coordinates(pc) for pc in P_0}


    def __repr__(self):
        output_string =  "Step "+str(self.k)+"\n" \
                        "Current time       t_"+str(self.k)+": "+str(self.t_k)+"\n"\
                        "Current location   c_" + str(self.k) + ": " + str(self.c_k) + "\n"\
                        "Remaining deliv    D_" + str(self.k) + ": " + str(self.D_k) + "\n"\
                        "Remaining pickup   P_" + str(self.k) + ": " + str(self.P_k) + "\n"
        return output_string


class PerfTracker:
    """
    To keep track of values and any data generated by the simulation
    """
    def __init__(self, simulator):
        self.simulator = simulator

        # time t_k
        self.time_t_k = [self.simulator.state.t_k]

        # nb of total known jobs at cur step
        self.total_nb_known_jobs_k = [len(self.simulator.served_deliv_pc) + len(self.simulator.served_pickup_pc) \
        + len(self.simulator.state.P_k) + len(self.simulator.state.D_k)]

        # total nb of served jobs at each step - deliv and pickups
        self.nb_served_jobs_k = [len(self.simulator.served_deliv_pc) + len(self.simulator.served_pickup_pc)]

        # value of the objective function at each step
        self.obj_func_k = [len(self.simulator.served_deliv_pc)*0.5 + len(self.simulator.served_pickup_pc)*1]

        # total distance since start
        self.total_distance = [0]

        # total waiting time since start
        self.total_waiting_periods = 0

        # average time between two deliveries/pickups OR average deliv/pickups per hour
        elapsed_time = self.simulator.state.t_k - self.simulator.departure_time
        self.average_deliv_h = []
        self.average_pickup_h = []

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
        self.total_nb_known_jobs_k.append(len(simulator.served_deliv_pc) + len(simulator.served_pickup_pc) \
        + len(simulator.state.P_k) + len(simulator.state.D_k))

        # nb of served jobs at each step - deliv and pickups
        self.nb_served_jobs_k.append(len(simulator.served_deliv_pc) + len(simulator.served_pickup_pc))

        # value of the objective function at each step
        self.obj_func_k.append(len(simulator.served_deliv_pc)*0.5 + len(simulator.served_pickup_pc)*1)

        # total distance since start
        self.total_distance.append(self.total_distance[-1] + distance)

        # total waiting time since start
        if decision == 0:
            self.total_waiting_periods += 1

        # average time between two deliveries/pickups OR average deliv/pickups per hour
        elapsed_time = simulator.state.t_k - simulator.departure_time
        self.average_deliv_h.append(len(simulator.served_deliv_pc) / (elapsed_time.total_seconds()/60))
        self.average_pickup_h.append(len(simulator.served_pickup_pc) / (elapsed_time.total_seconds()/60))

        # average time between two deliveries

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
    def __init__(self, scenario_mode="random", nb_truck=15, nb_deliv_jobs=10, nb_pickup_jobs=10, seed=10):

        self.nb_deliv_jobs = nb_deliv_jobs
        self.nb_pickup_jobs = nb_pickup_jobs

        self.departure_location = 119114
        self.departure_coord = [1.2805405, 103.7863179]

        # BASE SETTINGS
        if scenario_mode == "random":
            self.data, self.data_indices, _,self.sorted_truck, self.idx_to_pc, self.pc_to_idx \
                = base_data(nb_truck=nb_truck)

            # SCENARIO GENERATOR
            # pickup jobs
            # pickup location candidates
            data = self.data[self.data["ReadyTimePickup"] != 0]  # work on pickup data only
            pickup_candidates = [int(pc) for pc in self.idx_to_pc if len(data[data["PostalCode"] == pc]) > 0]

            # selection of pickups
            self.pc_pickup_jobs = random.sample(pickup_candidates, nb_pickup_jobs)

            # delivery jobs among remaining locations (arrangement)
            # selection of deliveries
            random.seed(seed)
            self.pc_deliv_jobs = random.sample(set(self.idx_to_pc) - set(self.pc_pickup_jobs), nb_deliv_jobs)

        elif scenario_mode == "realistic":
            self.data, _, self.data_scenario, _, self.idx_to_pc, self.pc_to_idx \
                = base_data(nb_truck=nb_truck)

            print("SELF.IDX_TO_PC 414 ", self.idx_to_pc[414])
            print("SELF.IDX_TO_PC 415 ", self.idx_to_pc[415])
            # SCENARION GENERATOR
            # deliveries locations: set as once visited it is served
            self.pc_deliv_jobs = \
                list(set(self.data_scenario[self.data_scenario["ReadyTimePickup"] == 0]["PostalCode"].values))

            # pickup locations: not a set as there can be repeated location with different time
            self.pc_pickup_jobs = \
                list(self.data_scenario[self.data_scenario["ReadyTimePickup"] != 0]["PostalCode"].values)
        else:
            raise ValueError("In Simulator object creation, please choose a valid scenario mode. You input "
                             +str(scenario_mode))

        # dict containing the jobs
        self.all_jobs_pc_to_coord = {}
        self.all_jobs_pc_to_coord["pickups"] = {pc:get_coordinates(pc) for pc in self.pc_pickup_jobs}
        self.all_jobs_pc_to_coord["deliveries"] = {pc:get_coordinates(pc) for pc in self.pc_deliv_jobs}
        # print("dict of all jobs ", self.all_jobs_pc_to_coord)

        self.all_jobs_coord_to_pc = {}
        self.all_jobs_coord_to_pc["pickups"] = {coord:pc for pc, coord in self.all_jobs_pc_to_coord["pickups"].items()}
        self.all_jobs_coord_to_pc["deliveries"] = {coord: pc for pc, coord in self.all_jobs_pc_to_coord["deliveries"].items()}

        self.pc_to_coord = {}
        for type_job in ["pickups", "deliveries"]:
            for pc, coord in self.all_jobs_pc_to_coord[type_job].items():
                self.pc_to_coord[pc] = coord
        self.pc_to_coord[self.departure_location] = self.departure_coord

        # TODO Pb ! we must re-create realistic scenarios. This part is just a headstart
        # when
        if scenario_mode == "random":
            pickup_arrival_time_mean, pickup_arrival_time_var = self.get_ready_time()
            self.pickup_loc_to_time = {
                location: datetime.datetime(
                    year=1970,
                    month=1,
                    day=1,
                    hour=0,
                    minute=0,
                    second=0,
                    microsecond=0,
                    tzinfo=None)
                    +
                    datetime.timedelta(seconds=math.floor(random.gauss(
                        pickup_arrival_time_mean[int(location)],
                        pickup_arrival_time_var[int(location)]))
                    )
                for location in self.pc_pickup_jobs
            }
        elif scenario_mode == "realistic":

            self.pickup_loc_to_time = defaultdict(list)

            for idx, row in self.data_scenario[self.data_scenario["ReadyTimePickup"] != 0].iterrows():
                self.pickup_loc_to_time[row["PostalCode"]].append(
                    datetime.datetime.strptime(str(int(row["StopStartTime"])), '%H%M').replace(
                        year=1970,
                        month=1,
                        day=1,
                        tzinfo=None)
                )

        else:
            raise ValueError("In Simulator object creation, please choose a valid scenario mode. You input "
                             +str(scenario_mode))

        self.pickup_time_to_loc = {}
        for loc, time in self.pickup_loc_to_time.items():
            for i in time:
                self.pickup_time_to_loc[i] = loc

        # Keep history of states with a list of State object
        self.served_pickup_pc = []
        self.served_deliv_pc = []

        # State
        self.departure_time = datetime.datetime(year=1970,
                                             month=1,
                                             day=1,
                                             hour=8,
                                             minute=0)

        # Initialize state by filtering unknown deliveries at departure time
        # TODO Solve this question of the departure location ! For now, 1 chosen among the deliveries
        self.state = State(initial_location=self.pc_deliv_jobs[0],  # self.departure_location,
                           initial_time=self.departure_time,
                           P_0=[self.pickup_time_to_loc[t]
                                for t in filter_lower_datetime(self.departure_time,
                                                               [item for sublist in self.pickup_loc_to_time.values()
                                                                for item in sublist])
                                ],
                           D_0=self.pc_deliv_jobs[1:]
                           )
        # Perf tracker
        self.tracker = PerfTracker(self)

        # Policies
        # Train and retrieve the model
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
                                                                                              tzinfo=None)
                 - datetime.datetime.now().replace(
                    year=1970,
                    month=1,
                    day=1,
                    hour=0,
                    minute=0,
                    second=0,
                    microsecond=0,
                    tzinfo=None)
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
            del self.state.D_k[decision]  # decision has been served
            self.served_deliv_pc.append(decision)  # record the action

        # remaining pickups
        elif decision in list(self.state.P_k.keys()):
            # remove
            del self.state.P_k[decision]
            self.served_pickup_pc.append(decision)

            # update by looking at the pre-loaded scenario in self.pickup_loc_to_time, not already served
            new_jobs = [job for job, t in self.pickup_loc_to_time.items() if min(t) < self.state.t_k
                        and (len(t) > self.served_pickup_pc.count(job) or job not in self.served_pickup_pc)]
            for new_job in new_jobs:
                new_job_coord = get_coordinates(new_job)
                self.state.P_k[new_job] = new_job_coord

        elif decision == 0:  # wait case, no removing, just updating
            new_jobs = [job for job, t in self.pickup_loc_to_time.items() if min(t) < self.state.t_k
                        and (len(t) > self.served_pickup_pc.count(job) or job not in self.served_pickup_pc)]
            for new_job in new_jobs:
                new_job_coord = get_coordinates(new_job)
                self.state.P_k[new_job] = new_job_coord

        else:
            raise ValueError('Error in simulator\'s method update_remaining_jobs: ' \
                             'decision is not in self.state.D_k or self.state.P_k keys')

    def display_scenario(self):
        print("Scenario:")
        print("     Pickups")

        for loc, time in self.pickup_loc_to_time.items():
            print("         ", loc, time)

        print("____________")
        print("     Deliveries")

        for loc in self.pc_deliv_jobs:
            print("         ", loc)

    # --------
    #   Policies

    def nearest_neigbor(self, pc):
        """
        Return the closest place from pc among self.pc_deliv_jobs and self.pc_pickup_jobs
        :param pc: int, postal code
        :return: int, postal code. 0 if no more jobs
        """
        coord = get_coordinates(pc)
        # deliveries
        pdist_deliv = {haversine(coord[0], coord[1], pcoord[0], pcoord[1]):pc for pc, pcoord in self.state.D_k.items()}
        pdist_list_deliv = list(pdist_deliv.keys())
        if len(pdist_list_deliv) > 0:
            val_deliv_min = min(pdist_list_deliv)
        else:
            val_deliv_min = 1e6  # great value to be discarded when comparing with val_pickup_min
        # pickups
        pdist_pickup = {haversine(coord[0], coord[1], pcoord[0], pcoord[1]):pc for pc, pcoord in self.state.P_k.items()}
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
        Return the decision (postal code) of our trained heuristic among self.state.D_k or self.state.P_k
        :param: pc, int: postal code of the current location
        :return: int, postal code. 0 if no more jobs
        """
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
            raise ValueError('Problem with tasks, which is negative.')

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
            delta_time, distance = get_travel_time(self.state.c_k, decision, self.state.t_k)
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

    def run_simulator(self, noon_deadline, policy="nearest_neighbor", plot=initialize_map()):

        sim.display_scenario()

        print("start time ", sim.state.t_k)
        while self.state.t_k < noon_deadline:
            # print state
            print(self.state)

            print("-----------------------------")
            # print decision
            print("** DECISION ** ")
            if policy == "nearest_neighbor":
                decision = self.nearest_neigbor(sim.state.c_k)
            elif policy == "computer_driver_heuristic":
                decision = self.computer_driver_heuristic(sim.state.c_k)
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
        # TODO add a location (1, 1, ..., 1) in order to include the first locations in the labels
        # Simulator
        sim = Simulator(scenario_mode="realistic", nb_truck=5)
        noon_deadline = datetime.datetime(year=1970,
                                          month=1,
                                          day=1,
                                          hour=12,
                                          minute=0,
                                          second=0,
                                        microsecond=0)

        # decision = sim.computer_driver_heuristic(list(sim.state.D_k.keys())[0])
        # print("Decision made by the computer :" , decision)

        # Run
        plot, tracker = sim.run_simulator(noon_deadline=noon_deadline,
                                 policy="nearest_neighbor")

        # Tracker info
        print(tracker)

        # Map
        show_map(plot=plot)

        # TODO: insert one of the scenario used to train the computer heuristic
        # TODO: try to improve the travel time function

        # LATER...
        # TODO improve coordinates of our postal codes
        # TODO improve travel time function
        # Present info
