import numpy as np
import networkx as nx

NUM_GOODS = 25
NUM_BIDS = 100
MAX_BIDS_PER_BIDDER = 5
THREE_PROB = 1.0
ADDITIONAL_NEIGHBOR_PROB = 0.2
JUMP_PROB = 0.1
DEVIATION = 0.5
MAX_GOOD_VALUE = 100.0
ADDITIONAL_GOOD_PROB = 0.9
ADDITIVITY = 0.2
BUDGET_FACTOR = 1.5
RESALE_FACTOR = 0.5

class Proximity:
    """
    Generates a Combinatorial Auction problem following the 'proximity' scheme found in section 4.2 of
        Kevin Leyton-Brown, Mark Pearson, and Yoav Shoham. (2000).
        Towards a universal test suite for combinatorial auction algorithms.
        Proceedings of ACM Conference on Electronic Commerce (EC-00) 66-76.

    Parameters
    ----------
    num_goods : int
        A lower bound on the number of goods to be bid on. The true num_goods will be the next highest square number.
        Default: 25
    num_bids : int
        The total number of bids across all bidders.
        Default: 100
    max_bids_per_bidder : int
        For each bidder, the maximum number of bids they can submit.
        Default: 5
    three_prob : float
        The probability a non-edge node will only be connected to three adjacent nodes.
        Default: 1.0
    additional_neighbor_prob : float
        The probability a diagonal edge will be added for a non-edge node.
        Default: 0.2
    jump_prob : float
        The probability a random node, as opposed to an adjacent node, will be added to a bundle.
        Default: 0.05
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

    def __init__(self, seed):
        self.ingest_parameters()

        self.name = "Proximity"

        self.seed = seed
        np.random.seed(self.seed)

        self.instantiate_graph()
        self.num_items = len(self.graph.nodes())

        self.instantiate_distribution()
        self.num_bidders = len(self.XOR_bids)

        print(f"Distribution: {self.name}, Num Bidders: {self.num_bidders}, Num Items: {self.num_items}, Seed: {self.seed}")

    def ingest_parameters(self):
        self.num_goods = NUM_GOODS
        self.num_bids = NUM_BIDS
        self.max_bids_per_bidder = MAX_BIDS_PER_BIDDER
        self.three_prob = THREE_PROB
        self.additional_neighbor_prob = ADDITIONAL_NEIGHBOR_PROB
        self.jump_prob = JUMP_PROB
        self.deviation = DEVIATION
        self.max_good_value = MAX_GOOD_VALUE
        self.additional_good_prob = ADDITIONAL_GOOD_PROB
        self.additivity = ADDITIVITY
        self.budget_factor = BUDGET_FACTOR
        self.resale_factor = RESALE_FACTOR

    def return_valid_neighbors(self, node, diagonal=False):
        neighbors = set()
        if diagonal:
            for i in [-1, 1]:
                for j in [-1, 1]:
                    if (node[0] + i, node[1]+j) in self.graph.nodes():
                        neighbors.add((node[0] + i, node[1]+j))
        else:
            for i in [-1, 1]:
                if (node[0] + i, node[1]) in self.graph.nodes():
                    neighbors.add((node[0] + i, node[1]))
                if (node[0], node[1] + i) in self.graph.nodes():
                    neighbors.add((node[0], node[1] + i))

        return neighbors

    def instantiate_graph(self):
        self.graph = nx.Graph()
        num_rows = int(np.ceil(np.sqrt(self.num_goods)))

        edge_nodes = set()
        for i in range(num_rows):
            for j in range(num_rows):
                self.graph.add_node((i,j), pos=(i, j))

                if i == 0 or i == num_rows-1 or j == 0 or j == num_rows-1:
                    edge_nodes.add((i,j))

        for node in self.graph.nodes():
            if node in edge_nodes:
                for neighbor in self.return_valid_neighbors(node, False):
                    self.graph.add_edge(node, neighbor)
            else:
                if np.random.uniform() < self.three_prob:
                    rand_neighbor_skip = np.random.randint(4)
                else:
                    rand_neighbor_skip = -1

                for j, neighbor in enumerate(self.return_valid_neighbors(node, False)):
                    if j != rand_neighbor_skip:
                        self.graph.add_edge(node, neighbor)

                while np.random.uniform() < self.additional_neighbor_prob:
                    valid_neighbors = list(self.return_valid_neighbors(node, False))
                    rand_diag_neighbor = valid_neighbors[np.random.choice(range(len(valid_neighbors)))]
                    # add diagonal edge if it doesn't cross an exist diagonal edge
                    if ((node[0],rand_diag_neighbor[1]),(rand_diag_neighbor[0], node[1])) not in self.graph.edges():
                        self.graph.add_edge(node, rand_diag_neighbor)


    def add_good_to_bundle(self, bundle, private_values_normalized):
        remaining_items = list(set(self.graph.nodes()) - bundle)
        if np.random.uniform() < self.jump_prob:
            item_idx = np.random.choice(range(len(remaining_items)))
        else:
            s = 0
            probs = np.zeros(len(remaining_items))
            for rem_item in remaining_items:
                for included_item in bundle:
                    if self.graph.has_edge(included_item, rem_item):
                        index = list(self.graph.nodes()).index(rem_item)
                        s += private_values_normalized[index]

                        index2 = remaining_items.index(rem_item)
                        probs[index2] += private_values_normalized[index]
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
            if len(substitutable_bids) > self.max_bids_per_bidder:
                substitutable_bids = dict(sorted(substitutable_bids.items(), key=lambda item: item[1])[:self.max_bids_per_bidder])

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