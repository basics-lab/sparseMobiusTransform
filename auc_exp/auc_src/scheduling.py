import numpy as np

NUM_SLOTS = 400
NUM_BIDS = 5  # we only need to generate for the first bidder, who has at most 5 bids
DEVIATION = 0.5
ADDITIVITY = 0.2
ADDITIONAL_DEADLINE_PROB = 0.9
MAX_BIDS_PER_BIDDER = 5

class Scheduling:
    """
    Generates a Combinatorial Auction problem following the 'temporal scheduling' scheme found in section 4.5 of
        Kevin Leyton-Brown, Mark Pearson, and Yoav Shoham. (2000).
        Towards a universal test suite for combinatorial auction algorithms.
        Proceedings of ACM Conference on Electronic Commerce (EC-00) 66-76.

    Parameters
    ----------
    num_slots : int
        The total number of slots to be scheduled.
        Default: 10
    num_bids : int
        The total number of bids submitted.
        Default: 100
    deviation : float
        The random deviation in value for a bidder.
        Default: 0.5
    additivity : float
        The scaling of how valuable a job is based on its length. Note that additivity < 0 gives subadditivity.
        Default: 0.2
    additional_deadline_prob : float
        The probability another deadline will be considered.
        Default: 0.9
    max_bids_per_bidder : int
        For each bidder, the maximum number of bids they can submit.
        Default: 5
    """

    def __init__(self, seed, regime='small'):
        if regime == 'small':
            self.num_slots = 20
        else:
            self.num_slots = 400
        self.num_items = self.num_slots
        self.ingest_parameters()
        self.num_items = self.num_slots
        self.name = "Temporal Scheduling"

        self.seed = seed
        np.random.seed(self.seed)

        self.instantiate_distribution()
        self.num_bidders = len(self.XOR_bids)

        # print(f"Distribution: {self.name}, Num Bidders: {self.num_bidders}, Num Items: {self.num_items}, Seed: {self.seed}")

    def ingest_parameters(self):
        self.num_bids = NUM_BIDS
        self.deviation = DEVIATION
        self.additivity = ADDITIVITY
        self.additional_deadline_prob = ADDITIONAL_DEADLINE_PROB
        self.max_bids_per_bidder = MAX_BIDS_PER_BIDDER

    def instantiate_distribution(self):
        self.XOR_bids = {}
        i = 0
        tot_bids = 0
        while tot_bids < self.num_bids:
            job_length = np.random.randint(1, self.num_slots)
            job_deadline = np.random.randint(job_length, self.num_slots)
            job_start = np.random.randint(0, job_deadline - job_length + 1)
            dev = np.random.uniform(low=1-self.deviation, high=1+self.deviation)
            cur_max_deadline = 0
            new_deadline = job_deadline

            # Always create first bids with original deadline
            for start in range(job_start, self.num_slots - job_length):
                end = start + job_length
                if new_deadline >= end > cur_max_deadline:
                    bid = dev * (job_length ** (1 + self.additivity)) * job_deadline / new_deadline

                    if i in self.XOR_bids:
                        self.XOR_bids[i][(start, start + job_length)] = bid

                        if len(self.XOR_bids[i]) >= self.max_bids_per_bidder:
                            break
                    else:
                        self.XOR_bids[i] = {(start, start + job_length): bid}
            cur_max_deadline = new_deadline
            if cur_max_deadline == self.num_slots - 1:
                pass
            else:
                new_deadline = np.random.randint(cur_max_deadline + 1, self.num_slots)

                if i in self.XOR_bids:
                    # Create additional bids at random
                    while np.random.uniform() < self.additional_deadline_prob and len(self.XOR_bids[i]) < self.max_bids_per_bidder:
                        for start in range(self.num_slots - job_length):
                            end = start + job_length
                            if new_deadline >= end > cur_max_deadline:
                                bid = dev * (job_length ** (1 + self.additivity)) * job_deadline / new_deadline
                                # this would be a smaller bid if repeated from above, so we check if bid exists.
                                if (start, start + job_length) not in self.XOR_bids[i] and len(self.XOR_bids[i]) < self.max_bids_per_bidder:
                                    self.XOR_bids[i][(start, start + job_length)] = bid
                        cur_max_deadline = new_deadline
                        if cur_max_deadline == self.num_slots - 1:
                            break
                        else:
                            new_deadline = np.random.randint(cur_max_deadline + 1, self.num_slots)
            if i in self.XOR_bids:
                tot_bids += len(self.XOR_bids[i])
                i += 1

    def value_function(self, bidder_id, allocation):
        slots_allocated_i = set()
        for j in range(self.num_items):
            if allocation[j] == bidder_id + 1:
                slots_allocated_i.add(j)

        value = 0
        # Find the XOR bids for the bidder
        XOR_bids_i = self.XOR_bids[bidder_id]
        for slot_pair, val in XOR_bids_i.items():
            if slot_pair[0] in slots_allocated_i and slot_pair[1] in slots_allocated_i:
                if val > value:
                    value = val
        return value