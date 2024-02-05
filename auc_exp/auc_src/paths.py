import numpy as np
import networkx as nx
from itertools import islice

NUM_BIDS = 5  # we only need to generate for the first bidder, who has at most 5 bids
INITIAL_CONNECTIONS = 2
BUILDING_PENALTY = 1.7
SHIPPING_COST_FACTOR = 1.5
MAX_BIDS_PER_BIDDER = 5

class Paths:
    """
    Generates a Combinatorial Auction problem following the 'paths' scheme found in section 4.1 of
        Kevin Leyton-Brown, Mark Pearson, and Yoav Shoham. (2000).
        Towards a universal test suite for combinatorial auction algorithms.
        Proceedings of ACM Conference on Electronic Commerce (EC-00) 66-76.

    Parameters
    ----------
    num_bids : int
        The max number of XOR bids for all bidders.
        Default: 100
    num_cities : int
        The number of nodes in the graph.
        Default: 10
    initial_connections : int
        The number of edges per city in the initial graph.
        Default: 2
    num_building_paths : int
        The number of paths used to add additional non-nearest neighbor edges.
        Default: 10
    building_penalty : float
        A penalty for adding additional edges to the graph not included in initial_connections.
        Default: 1.7
    shipping_cost_factor : float
        A coefficient on the revenue achievable by connecting two cities.
        Default: 1.5
    max_bids_per_bidder : int
        Max number of bids per bidder.
        Default: 5
    """

    def __init__(self, seed, regime='small'):
        if regime == 'small':
            self.num_cities = 15
            self.target_edges = 20
            self.shipping_cost_factor = 2.3  # too conservative in original paper for small regime, prevents substitutable paths
        else:
            self.num_cities = 275
            self.target_edges = 400
            self.shipping_cost_factor = 1.5

        self.ingest_parameters()
        self.name = "Paths"

        self.seed = seed
        np.random.seed(self.seed)
        self.instantiate_graph()
        self.num_items = len(self.graph.edges)
        self.instantiate_distribution()
        self.num_bidders = len(self.XOR_bids)
        print(self.XOR_bids[0])

        # print(f"Distribution: {self.name}, Num Bidders: {self.num_bidders}, Num Items: {self.num_items}, Seed: {self.seed}")

    def ingest_parameters(self):
        self.num_bids = NUM_BIDS
        self.initial_connections = INITIAL_CONNECTIONS
        self.building_penalty = BUILDING_PENALTY
        self.max_bids_per_bidder = MAX_BIDS_PER_BIDDER

    def instantiate_graph(self):
        self.graph = nx.Graph()
        positions = np.random.uniform(size=(self.num_cities, 2))
        self.graph.add_nodes_from(range(self.num_cities), pos=positions)

        self.distances_sq = np.sum((positions[:, np.newaxis, :] - positions[np.newaxis, :, :]) ** 2, axis=-1)
        for i in range(self.num_cities):
            # Find the self.initial_connections closest cities to city i
            closest_cities = np.argsort(self.distances_sq[i])[1:self.initial_connections+1]
            for j in closest_cities:
                if len(self.graph.edges()) < self.target_edges:
                    self.graph.add_edge(i, j, weight=np.sqrt(self.distances_sq[i][j]))

        # Add additional edges to the graph
        count = 0
        while len(self.graph.edges()) < self.target_edges and count < 100:
            C = self.graph.copy()
            # Add missing edges with additional building penalty to C
            for j in range(self.num_cities):
                for k in range(self.num_cities):
                    if j != k and not C.has_edge(j, k):
                        C.add_edge(j, k, weight=self.building_penalty * np.sqrt(self.distances_sq[j][k]))

            # Pick two random cities
            city1, city2 = np.random.choice(self.num_cities, size=2, replace=False)
            shortest_path = nx.shortest_path(C, city1, city2, weight='weight')
            for l in range(len(shortest_path) - 1):
                n1, n2 = shortest_path[l], shortest_path[l + 1]
                if not self.graph.has_edge(n1, n2) and len(self.graph.edges()) < self.target_edges:
                    self.graph.add_edge(n1, n2, weight=np.sqrt(self.distances_sq[n1][n2]))
            count += 1

        if count == 100:
            # Failed to generate graph with target edges, retry with different node positions
            self.instantiate_graph()

        # Give IDs to each edge
        self.edge_ids = {edge: i for i, edge in enumerate(self.graph.edges())}

        if not nx.is_connected(self.graph):
            self.instantiate_graph()



    def instantiate_distribution(self):
        self.XOR_bids = {}
        i = 0
        tot_bids = 0
        while tot_bids < self.num_bids:

            # Randomly select a pair of cities
            city1, city2 = np.random.choice(self.num_cities, size=2, replace=False)
            # Generate a random revenue coefficient
            revenue_coefficient = np.random.uniform(low=1, high=self.shipping_cost_factor)
            city_distance = np.sqrt(self.distances_sq[city1][city2])

            # Find the max_bid_set_size cheapest paths
            cheapest_paths = list(islice(nx.shortest_simple_paths(self.graph, source=city1, target=city2, weight='weight'), self.max_bids_per_bidder))
            path_weights = [nx.path_weight(self.graph, path, 'weight') for path in cheapest_paths]

            bids = {}
            for path_id in range(len(cheapest_paths)):
                if path_weights[path_id] <= revenue_coefficient * city_distance:
                    value = (revenue_coefficient * city_distance) - path_weights[path_id]
                    path_edge_ids = []
                    # add the required edges to the XOR bid
                    for l in range(len(cheapest_paths[path_id]) - 1):
                        n1, n2 = cheapest_paths[path_id][l], cheapest_paths[path_id][l + 1]
                        if n1 < n2:
                            path_edge_ids.append(self.edge_ids[(n1, n2)])
                        else:
                            path_edge_ids.append(self.edge_ids[(n2, n1)])
                    bids[tuple(path_edge_ids)] = value
            if len(bids) > 0:
                self.XOR_bids[i] = bids
                i += 1
                tot_bids += len(bids)

    def value_function(self,  bidder_id, allocation):
        # Use XOR bidding lanauge to compute value for bidder i
        edges_allocated_i = set()
        for j in range(self.num_items):
            if allocation[j] == bidder_id + 1:
                edges_allocated_i.add(j)

        value = 0
        # Find the XOR bids for the bidder
        XOR_bids_i = self.XOR_bids[bidder_id]
        for path_edge_ids, val in XOR_bids_i.items():
            if set(path_edge_ids).issubset(edges_allocated_i):
                if val > value:
                    value = val
        return value
