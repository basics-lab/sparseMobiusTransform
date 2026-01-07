import sys
import os
import numpy as np
import pandas as pd
from smt.input_signal_subsampled import SubsampledSignal
from smt.smt import SMT
from smt.query import get_Ms_and_Ds
from smt.utils import dec_to_bin_vec, bin_vec_to_dec
from graph_utils import GraphAnalyzer
import time

# Add parent directory to path to import graph_utils if needed (though it seems to be in pythonpath or same dir structure)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class GraphSignal(SubsampledSignal):
    def __init__(self, graph_analyzer, **kwargs):
        self.graph_analyzer = graph_analyzer
        # Set required attributes for SubsampledSignal
        kwargs['n'] = graph_analyzer.num_nodes
        kwargs['q'] = 2
        if 'noise_sd' not in kwargs:
            kwargs['noise_sd'] = 0.0
        self.noise_sd = kwargs['noise_sd']
        super().__init__(**kwargs)

    def subsample(self, query_indices):
        """
        query_indices: a list of integers, where each integer represents a subset of nodes.
        Vectorized implementation for performance.
        """
        # 1. Fast conversion of indices to boolean mask matrix (n, batch_size)
        # We need the bit corresponding to node i to be at index n-1-i??
        # Wait, my previous manual loop: mask[self.n - 1 - i] = True for LSB i.
        # This maps LSB to n-1. MSB to 0.
        # fast_dec_to_bin_vec returns array where index 0 is MSB.
        # So fast_dec_to_bin_vec result matches the working bit order.
        
        n = self.n
        n_bytes = (n + 7) // 8
        byte_list = [int(val).to_bytes(n_bytes, byteorder='big') for val in query_indices]
        all_bytes = b''.join(byte_list)
        uint8_arr = np.frombuffer(all_bytes, dtype=np.uint8)
        bits = np.unpackbits(uint8_arr)
        bits = bits.reshape(len(query_indices), -1)
        
        # Take last n bits (corresponding to MSB at 0 if we consider n bits)
        # Wait, if n_bytes*8 > n.
        # 'big' endian: 1 -> ...00001.
        # unpackbits -> 0, ..., 0, 1.
        # We want the LAST n bits?
        # My check_bits.py showed that taking [:,-n:] matched dec_to_bin_vec.
        # dec_to_bin_vec puts MSB at 0.
        # So we use X = bits[:, -n:].T to get (n, batch).
        
        X = bits[:, -n:].T # (n, batch)
        
        # 2. Batch edge counting
        # A * X
        neighbors = self.graph_analyzer.adj_matrix.dot(X) # (n, batch)
        
        # Element-wise multiply and sum column-wise
        # This calculates x^T A x efficiently for each column x
        total = np.sum(neighbors * X, axis=0) # (batch,)
        
        return total // 2

    def get_MDU(self, ret_num_subsample, ret_num_repeat, b, trans_times=False):
        # We don't need to add noise for now, just delegate to super
        return super().get_MDU(ret_num_subsample, ret_num_repeat, b, trans_times)
        
def run_experiment():
    # Configuration
    data_file = 'soc-sign-bitcoinalpha.csv'
    
    # Check if file exists
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found.")
        return

    # To keep things fast for testing, let's subsample the graph
    # n=100 nodes should be enough to demonstrate
    # If n is too small, we might not have many edges.
    n_nodes = 100 
    
    print(f"Loading graph from {data_file} with first {n_nodes} nodes...")
    # We can pass subsample_ratio, but we want exactly n_nodes for consistent sizing.
    # GraphAnalyzer doesn't support 'n_nodes' directly, only ratio.
    # But we can assume the file is large, so let's just pick a ratio or modify GA?
    # Actually, let's just use the full graph and let the user know, 
    # OR better: use subsample_ratio to approximate, OR just rely on logic in GraphSignal to use n.
    # 
    # Wait, GraphSignal needs 'n' to match GraphAnalyzer.num_nodes.
    # Let's use a ratio that gives us roughly 100 nodes for speed.
    # The full graph might verify large.
    # Let's assume we want a controlled size.
    # The user said "use the test_graph.py script to check everything is working".
    # And "soc-sign-bitcoinalpha.csv" is the file.
    # Let's try to get a small chunk.
    # 
    # Hack: We can create a temporary file with just the nodes we want or subclass GraphAnalyzer.
    # But GraphAnalyzer has a subsample_ratio.
    # Let's try ratio=0.1 first, check size, if too big, we might want smaller.
    
    # Actually, better approach:
    # 1. Load full graph (or large chunk)
    # 2. Extract a subgraph of exactly size N explicitly if needed.
    # But GraphAnalyzer logic is:
    # if subsample_ratio < 1.0 -> subsamples.
    
    # Let's try with a small ratio.
    # soc-sign-bitcoinalpha has ~3k nodes? or 6k?
    # 3783 nodes.
    # 100 / 3783 ~= 0.026
    
    subsample_ratio = 0.5
    
    ga = GraphAnalyzer(data_file, subsample_ratio=subsample_ratio, seed=42)
    print(f"Graph loaded. Nodes: {ga.num_nodes}, Edges: {ga.adj_matrix.nnz // 2}")
    
    n = ga.num_nodes
    
    # SMT Parameters
    # As per quick_example.py
    q = 2
    # K is number of edges (non-zero coefficients).
    # We need q^b > K.
    # edges = 695. log2(695) ~ 9.4.
    # b = 10 or 11 should be enough.
    # Previous heuristic: log2(K) + 4 was too aggressive.
    # Let's try log2(K) + 1, ensuring min b=8.
    K = ga.adj_matrix.nnz // 2
    b = 12
    print(f"Calculated b: {b} (2^b = {2**b}) for K={K} edges.")
    
    query_args = {
        "query_method": "group_testing",
        "num_subsample": 3,
        "delays_method_source": "coded",
        "subsampling_method": "smt",
        "delays_method_channel": "identity",
        "num_repeat": 1,
        "b": b,
        "t": 4, # Degree 2 monomials, so t=2 is min, let's use 4 to be safe? Or 2? 
                # User said "monomials we want to learn are all degree 2".
                # If we know max degree is 2, t=2 is sufficient? 
                # "random_deg_t_vecs(t, n, x)" suggests t is related to some sparsity or degree check.
                # In group testing contexts, t often refers to number of defects.
                # Here, "defects" are the non-zero coeffs, i.e., edges connected to a node?
                # Actually, in 'coded' delays for source, t is often max hamming weight of indices.
                # Edges have hamming weight 2. So t=2 should be correct.
        "wt": np.log(2), # from quick_example parameter_set 3
        "p": 250
    }
    
    # Need a source decoder for "coded" delays
    # For H.W. 2, decode is simple?
    from smt.random_group_testing import decode_robust, decode
    
    # Since we are noiseless, maybe we don't need robust?
    # Parameter set 1 (Noiseless Low-Degree) uses delays_method_source="coded", source_decoder=decode
    
    # Let's stick to parameter_set 1 style since we are effectively noiseless (exact graph counts).
    
    qsft_args = {
        "num_subsample": 3,
        "num_repeat": 1,
        "reconstruct_method_source": "coded",
        "reconstruct_method_channel": "identity",
        "b": b,
        "noise_sd": 0,
        "source_decoder": decode 
    }
    
    print("Initializing GraphSignal...")
    signal = GraphSignal(ga, query_args=query_args)
    
    print("Running QSFT Transform...")
    sft = SMT(**qsft_args)
    
    start_time = time.time()
    result = sft.transform(signal, verbosity=2, report=True)
    end_time = time.time()
    
    print(f"QSFT complete in {end_time - start_time:.2f}s")
    
    locations = result.get("locations")
    print(f"Found {len(locations)} non-zero coefficients.")
    
    # Verification
    # Convert locations (tuples of bits? or something else?)
    # smt.py says: loc is list of tuples.
    # The tuples represent the binary vector k.
    
    true_edges = 0
    recovered_edges = 0
    true_positives = 0
    
    # Get true edges set
    # ga.adj_matrix is CSR.
    # We can iterate it.
    # Since it is symmetric, we only care about i < j.
    
    adj = ga.adj_matrix.tocoo()
    true_edge_set = set()
    for u, v in zip(adj.row, adj.col):
        if u < v:
            true_edge_set.add((u, v))
            
    # Process recovered locations
    recovered_edge_set = set()
    for loc in locations:
        # loc is a tuple representing the binary vector.
        # Check hamming weight
        loc_arr = np.array(loc).flatten() # it might be (n, 1) or (n,)
        indices = np.where(loc_arr > 0)[0]
        
        if len(indices) == 2:
            u, v = sorted(indices)
            recovered_edge_set.add((u, v))
        else:
            # print(f"Found non-edge term with degree {len(indices)}: {indices}")
            pass
            
    print(f"True Edges: {len(true_edge_set)}")
    print(f"Recovered Edges: {len(recovered_edge_set)}")
    
    intersection = true_edge_set.intersection(recovered_edge_set)
    print(f"Correctly Recovered: {len(intersection)}")
    
    if len(true_edge_set) > 0:
        recall = len(intersection) / len(true_edge_set)
        print(f"Recall: {recall:.4f}")
    
    if len(recovered_edge_set) > 0:
        precision = len(intersection) / len(recovered_edge_set)
        print(f"Precision: {precision:.4f}")
        
    print("\n--- DEBUG INFO ---")
    print("First 5 True Edges:", list(true_edge_set)[:5])
    print("First 5 Recovered Edges:", list(recovered_edge_set)[:5])
    if len(recovered_edge_set) > 0:
        sample_rec = list(recovered_edge_set)[0]
        # Check if they look like indices?

        
    # Check NMSE if curious (should be 0 for perfect recovery)
    # But getting the weights requires looking at gwht values.
    # We are just checking structure.

if __name__ == "__main__":
    run_experiment()
