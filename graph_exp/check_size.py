import numpy as np
from graph_utils import GraphAnalyzer

data_file = 'soc-sign-bitcoinalpha.csv'
subsample_ratio = 0.4
ga = GraphAnalyzer(data_file, subsample_ratio=subsample_ratio, seed=42)
num_edges = ga.adj_matrix.nnz // 2
print(f"Nodes: {ga.num_nodes}")
print(f"Edges: {num_edges}")

b = int(np.ceil(np.log2(num_edges + 1))) + 4
print(f"Calculated b: {b}")
print(f"2^b: {2**b}")
