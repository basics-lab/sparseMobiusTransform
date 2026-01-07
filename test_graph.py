import numpy as np
import pandas as pd
from graph_utils import GraphAnalyzer
import time

def create_toy_data():
    # 0 -- 1
    # |  /
    # 2 -- 3
    # 4 (isolated logic, let's say 4 is connected to 0 for fun)
    
    # Let's make a specific simple graph:
    # 0-1, 1-2, 2-0 (Triangle 0,1,2)
    # 2-3 (Edge sticking out)
    # So nodes are 0,1,2,3.
    # Edges: (0,1), (1,2), (2,0), (2,3)
    
    data = [
        [0, 1, 1, 0],
        [1, 2, 1, 0],
        [2, 0, 1, 0],
        [2, 3, 1, 0]
    ]
    df = pd.DataFrame(data, columns=['SOURCE', 'TARGET', 'RATING', 'TIME'])
    df.to_csv('toy_graph.csv', index=False, header=False)

def test_toy_graph():
    print("Testing Toy Graph...")
    create_toy_data()
    ga = GraphAnalyzer('toy_graph.csv')
    
    # Test 1: S = {0, 1, 2} (The triangle)
    # Edges inside: (0,1), (1,2), (2,0) -> 3 edges
    # Cut: (2,3) -> 1 edge (connects 2 in S to 3 in not S)
    
    mask = np.zeros(ga.num_nodes, dtype=bool)
    # We don't know the mapping exactly, but since input IDs are 0,1,2,3 and they are sorted, 
    # the mapping should be identity.
    mask[[0, 1, 2]] = True 
    
    edges_inside = ga.get_subgraph_edge_count(mask)
    cut_size = ga.get_cut_size(mask)
    
    print(f"Subset {{0,1,2}}: Edges Inside = {edges_inside} (Expected 3), Cut Size = {cut_size} (Expected 1)")
    assert edges_inside == 3
    assert cut_size == 1

    # Test 2: S = {0}
    # Edges inside: 0
    # Cut: (0,1), (0,2) -> 2 edges
    mask = np.zeros(ga.num_nodes, dtype=bool)
    mask[0] = True
    
    edges_inside = ga.get_subgraph_edge_count(mask)
    cut_size = ga.get_cut_size(mask)
    
    print(f"Subset {{0}}: Edges Inside = {edges_inside} (Expected 0), Cut Size = {cut_size} (Expected 2)")
    assert edges_inside == 0
    assert cut_size == 2

    print("Toy Graph Tests Passed!\n")

def test_real_data():
    print("Testing Real Data soc-sign-bitcoinalpha.csv...")
    start_time = time.time()
    ga = GraphAnalyzer('soc-sign-bitcoinalpha.csv', subsample_ratio=1.0)
    load_time = time.time() - start_time
    print(f"Graph loaded in {load_time:.4f} seconds.")
    print(f"Num Nodes: {ga.num_nodes}")
    print(f"Num Edges: {ga.adj_matrix.nnz // 2}")
    
    # Random mask test
    np.random.seed(42)
    mask = np.random.rand(ga.num_nodes) > 0.5
    
    t0 = time.time()
    edges_inside = ga.get_subgraph_edge_count(mask)
    t1 = time.time()
    cut_size = ga.get_cut_size(mask)
    t2 = time.time()
    
    print(f"Random Half-Split: Edges Inside = {edges_inside}, Cut Size = {cut_size}")
    print(f"Time for Subgraph Count: {t1-t0:.6f} s")
    print(f"Time for Cut Size: {t2-t1:.6f} s")
    
    # Subsample Test
    print("\nTesting Subsampling (0.1 ratio)...")
    ga_small = GraphAnalyzer('soc-sign-bitcoinalpha.csv', subsample_ratio=0.1)
    print(f"Subsampled Graph Nodes: {ga_small.num_nodes}")
    print(f"Subsampled Graph Edges: {ga_small.adj_matrix.nnz // 2}")


if __name__ == "__main__":
    test_toy_graph()
    test_real_data()
