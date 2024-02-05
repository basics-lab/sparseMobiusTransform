import numpy as np

NUM_BIDS = 5  # we only need to generate for the first bidder, who has at most 5 bids
MAX_AIRPORT_VALUE = 5.0
LONGEST_FLIGHT_LENGTH = 10.0
DEVIATION = 0.5
EARLY_TAKEOFF_DEVIATION = 1
LATE_TAKEOFF_DEVIATION = 2
EARLY_LAND_DEVIATION = 1
LATE_LAND_DEVIATION = 2
DELAY_COEFFICIENT = 0.9
AMOUNT_LATE_COEFFICIENT = 0.75
MAX_BIDS_PER_BIDDER = 5

class Matching:
    """
    Generates a Combinatorial Auction problem following the 'temporal matching' scheme found in section 4.4 of
        Kevin Leyton-Brown, Mark Pearson, and Yoav Shoham. (2000).
        Towards a universal test suite for combinatorial auction algorithms.
        Proceedings of ACM Conference on Electronic Commerce (EC-00) 66-76.

    Parameters
    ----------
    num_slots_per_airport : int
        The number of timeslots per airport.
        Default: 5
    num_bids : int
        The total number of bids submitted.
        Default: 100
    max_airport_value : float
        The maximum value of having a slot at an airport.
        Default: 5.0
    longest_flight_length : float
        The time length of the longest flight.
        Default: 10.0
    deviation : float
        The random deviation in value for a slot pairing.
        Default: 0.5
    early_takeoff_deviation : int
        The earliest a takeoff can deviate.
        Default: 1
    late_takeoff_deviation : int
        The latest a takeoff can deviate.
        Default: 2
    early_land_deviation : int
        The earliest a landing can deviate.
        Default: 1
    late_land_deviation : int
        The latest a landing can deviate.
        Default: 2
    delay_coefficent : float
        The coefficient by which a bid is scaled by delayed a flight is in-air as compared to flying fastest route.
        Default: 0.9
    amount_late_coefficient : float
        The coefficient by which a bid is scaled by how late a landing is.
        Default: 0.75
    """

    def __init__(self, seed, regime='small'):
        if regime == 'small':
            self.num_slots_per_airport = 5
        else:
            self.num_slots_per_airport = 100
        self.ingest_parameters()
        self.num_items = 4 * self.num_slots_per_airport
        self.name = "Temporal Matching"

        self.seed = seed
        np.random.seed(self.seed)

        self.instantiate_graph()

        self.instantiate_distribution()
        self.num_bidders = len(self.XOR_bids)

        # print(f"Distribution: {self.name}, Num Bidders: {self.num_bidders}, Num Items: {self.num_items}, Seed: {self.seed}")

    def ingest_parameters(self):
        self.num_bids = NUM_BIDS
        self.max_airport_value = MAX_AIRPORT_VALUE
        self.longest_flight_length = LONGEST_FLIGHT_LENGTH
        self.deviation = DEVIATION
        self.early_takeoff_deviation = EARLY_TAKEOFF_DEVIATION
        self.late_takeoff_deviation = LATE_TAKEOFF_DEVIATION
        self.early_land_deviation = EARLY_LAND_DEVIATION
        self.late_land_deviation = LATE_LAND_DEVIATION
        self.delay_coefficent = DELAY_COEFFICIENT
        self.amount_late_coefficient = AMOUNT_LATE_COEFFICIENT
        self.max_bids_per_bidder = MAX_BIDS_PER_BIDDER

    def instantiate_graph(self):
        # 0 is Chicago O'Hare, 1 is Washington Reagen, 2 is JFK, 3 is LaGuardia.
        self.cities = {0: {'x': -87.75, 'y': 41.98333333},
                       1: {'x': -77.03333333, 'y': 38.85},
                       2: {'x': -73.783333, 'y': 40.65},
                       3: {'x': -73.8666666, 'y': 40.76666666}}
        self.edges = np.ones((4, 4)) * -1
        self.edge_set = set()

        for i in range(4):
            for j in range(4):
                # compute distances between airports (not including JFK <-> LaGuardia)
                if i != j:
                    if (i != 2 or j != 3) and (j != 2 or i != 3):
                        x_dist = self.cities[i]['x'] - self.cities[j]['x']
                        y_dist = self.cities[i]['y'] - self.cities[j]['y']
                        self.edges[i][j] = np.sqrt(x_dist ** 2 + y_dist ** 2)
                        self.edge_set.add((i, j))

        self.max_length = np.max(self.edges)

    def instantiate_distribution(self):
        self.XOR_bids = {}
        # Generate base values for each airport
        airport_values = np.random.uniform(low=0, high=self.max_airport_value, size=4)

        i = 0
        tot_bids = 0

        while tot_bids < self.num_bids:
            # Select an edge uniformly at random
            edge = list(self.edge_set)[np.random.choice(range(len(self.edge_set)))]

            # Compute the distance between the two airports
            distance = self.edges[edge[0]][edge[1]]

            # Find how many time slots needed to operate this flight
            min_time_for_flight = int(self.num_slots_per_airport * distance / self.max_length)

            # Randomly select an intended time slot for the flight to take off
            if min_time_for_flight == self.num_slots_per_airport:
                intended_takeoff = 0
            else:
                intended_takeoff = np.random.randint(low=0, high=self.num_slots_per_airport - min_time_for_flight)

            # Compute a random value deviation for the flight
            value_deviation = np.random.uniform(low=1-self.deviation, high=1+self.deviation)

            # For all possible delays of this slot, compute bid.
            for takeoff_time in range(max(0, intended_takeoff - self.early_takeoff_deviation), \
                    min(self.num_slots_per_airport, intended_takeoff + self.late_takeoff_deviation)):
                for landing_time in range(takeoff_time+min_time_for_flight, \
                                          min(self.num_slots_per_airport, intended_takeoff+min_time_for_flight+self.late_land_deviation)):
                    # How much longer of a path must be taken to operate this flight?
                    amount_delayed = max(0, landing_time - takeoff_time - min_time_for_flight)

                    # How late does this slot land?
                    amount_late = min(0, landing_time - (intended_takeoff + min_time_for_flight))

                    # Compute the value of the slot pair
                    slot_pair_value = value_deviation * (airport_values[edge[0]] + airport_values[edge[1]]) * \
                        np.power(self.delay_coefficent, amount_delayed) * np.power(self.amount_late_coefficient, \
                                                                                   amount_late)

                    # Add it to the set of XOR bids for bidder i
                    if i in self.XOR_bids and len(self.XOR_bids[i]) < self.max_bids_per_bidder:
                        self.XOR_bids[i][(edge[0], edge[1], takeoff_time, landing_time)] = slot_pair_value
                    elif i not in self.XOR_bids:
                        self.XOR_bids[i] = {(edge[0], edge[1], takeoff_time, landing_time): slot_pair_value}

            if i in self.XOR_bids:
                tot_bids += len(self.XOR_bids[i])
                i += 1



    def value_function(self,  bidder_id, allocation):
        # Use XOR bidding lanauge to compute value for bidder i
        airport_slots_allocated_i = set()
        allocation_reshaped = allocation.reshape((4, self.num_slots_per_airport))
        for airport_i in range(4):
            for slot_j in range(self.num_slots_per_airport):
                if allocation_reshaped[airport_i][slot_j] == bidder_id + 1:
                    airport_slots_allocated_i.add((airport_i, slot_j))

        value = 0

        # Find the XOR bids for the bidder
        XOR_bids_i = self.XOR_bids[bidder_id]
        for slot_pair, val in XOR_bids_i.items():
            if (slot_pair[0], slot_pair[2]) in airport_slots_allocated_i and (slot_pair[1], slot_pair[3]) in airport_slots_allocated_i:
                if val > value:
                    value = val
        return value