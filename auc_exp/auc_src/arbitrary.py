import numpy as np
import networkx as nx

NUM_BIDS = 5  # we only need to generate for the first bidder, who has at most 5 bids
MAX_BIDS_PER_BIDDER = 5
DEVIATION = 0.5
MAX_GOOD_VALUE = 100.0
ADDITIONAL_GOOD_PROB = 0.9
ADDITIVITY = 0.2
BUDGET_FACTOR = 1.5
RESALE_FACTOR = 0.5


class Arbitrary:
    """
    Generates a Combinatorial Auction problem following the 'arbitrary.cfg' scheme found in section 4.3 of
        Kevin Leyton-Brown, Mark Pearson, and Yoav Shoham. (2000).
        Towards a universal test suite for combinatorial auction algorithms.
        Proceedings of ACM Conference on Electronic Commerce (EC-00) 66-76.

    Parameters
    ----------
    num_goods : int
        The number of goods to be bid on.
        Default: 25
    num_bids : int
        The total number of bids submitted.
        Default: 100
    max_bids_per_bidder : int
        The maximum number of bids per bidder.
        Default: 5
    deviation : float
        The random deviation in private value for a good.
        Default: 0.5
    max_good_value : float
        The highest value that can be assigned to a good.
        Default: 100.0
    additional_good_prob : float
        The probability another good will be added to a bundle.
        Default: 0.9
    additivity : float
        The scaling of how valuable a bundle is based on its size. Note that additivity < 0 gives subadditivity.
        Default: 0.2
    budget_factor : float
        How much budget is available for a substitutable bid based on the value of the original bundle.
        Default: 1.5
    resale_factor : float
        Used to scale a lowerbound on resale value of a substitutable bundle.
        Default: 0.5
    """

    def __init__(self, seed, regime='small'):
        if regime == 'small':
            self.num_goods = 20
        else:
            self.num_goods = 400
        self.ingest_parameters()

        self.name = "Arbitrary"

        self.seed = seed
        np.random.seed(self.seed)

        self.instantiate_graph()
        self.num_items = len(self.graph.nodes())

        self.instantiate_distribution()
        self.num_bidders = len(self.XOR_bids)
        print(self.XOR_bids[0])

        # print(f"Distribution: {self.name}, Num Bidders: {self.num_bidders}, Num Items: {self.num_items}, Seed: {self.seed}")

    def ingest_parameters(self):
        self.num_bids = NUM_BIDS
        self.max_bids_per_bidder = MAX_BIDS_PER_BIDDER
        self.deviation = DEVIATION
        self.max_good_value = MAX_GOOD_VALUE
        self.additional_good_prob = ADDITIONAL_GOOD_PROB
        self.additivity = ADDITIVITY
        self.budget_factor = BUDGET_FACTOR
        self.resale_factor = RESALE_FACTOR

    def instantiate_graph(self):
        self.graph = nx.Graph()
        for i in range(self.num_goods):
            self.graph.add_node(i)

        for i in range(self.num_goods):
            for j in range(i+1, self.num_goods):
                self.graph.add_edge(i, j, weight=np.random.uniform())


    def add_good_to_bundle(self, bundle, private_values_normalized):
        remaining_items = list(set(self.graph.nodes()) - bundle)
        probs = np.zeros(len(remaining_items))
        s = 0
        for rem_item in remaining_items:
            for included_item in bundle:
                index = list(self.graph.nodes()).index(rem_item)
                s += self.graph[rem_item][included_item]["weight"] * private_values_normalized[index]

                index2 = remaining_items.index(rem_item)
                probs[index2] += self.graph[rem_item][included_item]["weight"] * private_values_normalized[index]
        item_idx = np.random.choice(range(len(remaining_items)), p=probs/s)
        new_bundle = bundle.copy()
        new_bundle.add(remaining_items[item_idx])
        return new_bundle

    def instantiate_distribution(self):
        self.XOR_bids = {}
        common_vals = np.random.uniform(low=1, high=self.max_good_value, size=len(self.graph.nodes()))
        i = 0
        tot_bids = 0
        while tot_bids < self.num_bids:
            valid_bundle = False
            while not valid_bundle:
                private_values = np.random.uniform(low=self.max_good_value - self.deviation, high=self.max_good_value + self.deviation, size=self.num_goods)
                private_values_scaled = (private_values + (self.deviation * self.max_good_value)) / (2*self.deviation * self.max_good_value)
                private_values_normalized = private_values_scaled / np.sum(private_values_scaled)

                # add a single good to the bundle weigted by the private value
                bundle = set()
                item_idx = np.random.choice(range(len(self.graph.nodes())), p=private_values_normalized)
                bundle.add(list(self.graph.nodes())[item_idx])

                while np.random.uniform() < self.additional_good_prob and len(bundle) < self.num_goods:
                    bundle = self.add_good_to_bundle(bundle, private_values_normalized)

                bundle_value = 0
                for item in bundle:
                    index = list(self.graph.nodes()).index(item)
                    bundle_value += private_values[index] + common_vals[index]
                bundle_value *= len(bundle) ** (1+self.additivity)

                if bundle_value > 0:
                    valid_bundle = True

            self.XOR_bids[i] = {tuple(bundle): bundle_value}

            # construct substitutable bids
            budget = self.budget_factor * bundle_value
            common_bundle_value = 0
            for item in bundle:
                index = list(self.graph.nodes()).index(item)
                common_bundle_value += common_vals[index]
            min_resale_value = self.resale_factor * common_bundle_value
            substitutable_bids = {}
            for item in bundle:
                subs_bundle = set()
                subs_bundle.add(item)

                while len(subs_bundle) < len(bundle):
                    subs_bundle = self.add_good_to_bundle(subs_bundle, private_values_normalized)

                subs_bundle_resale_val = 0
                subs_bundle_personal_val = 0
                for item2 in subs_bundle:
                    index = list(self.graph.nodes()).index(item2)
                    subs_bundle_resale_val += common_vals[index]

                    subs_bundle_personal_val += private_values[index] + common_vals[index]
                subs_bundle_personal_val *= len(subs_bundle) ** (1+self.additivity)

                if subs_bundle_resale_val >= min_resale_value and subs_bundle_personal_val <= budget:
                    substitutable_bids[tuple(subs_bundle)] = subs_bundle_personal_val

            # take self.max_substitutable_bids of the substitutable bids
            if len(substitutable_bids) > self.max_bids_per_bidder - 1:
                substitutable_bids = dict(sorted(substitutable_bids.items(), key=lambda item: item[1])[:self.max_bids_per_bidder - 1])

            for subs_bundle, subs_bundle_personal_val in substitutable_bids.items():
                self.XOR_bids[i][subs_bundle] = subs_bundle_personal_val

            tot_bids += len(self.XOR_bids[i])
            i += 1

    def value_function(self,  bidder_id, allocation):
        # Use XOR bidding lanauge to compute value for bidder i
        goods_allocated_i = set()
        for j in range(self.num_items):
            if allocation[j] == bidder_id + 1:
                good = list(self.graph.nodes())[j]
                goods_allocated_i.add(good)

        value = 0
        # Find the XOR bids for the bidder
        XOR_bids_i = self.XOR_bids[bidder_id]
        for good_ids, val in XOR_bids_i.items():
            if set(good_ids).issubset(goods_allocated_i):
                if val > value:
                    value = val
        return value