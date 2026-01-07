import numpy as np
import pandas as pd
from scipy import sparse

class GraphAnalyzer:
    def __init__(self, filename, subsample_ratio=1.0, seed=42):
        """
        Initializes the GraphAnalyzer with data from the given filename.
        
        Args:
            filename: Path to the CSV file.
            subsample_ratio: Fraction of nodes to keep (0.0 < ratio <= 1.0).
            seed: Random seed for reproducibility.
        """
        # Read data without header, assuming SOURCE, TARGET, RATING, TIME format
        df = pd.read_csv(filename, header=None, names=['SOURCE', 'TARGET', 'RATING', 'TIME'])
        
        # Extract edge list
        sources = df['SOURCE'].values
        targets = df['TARGET'].values
        
        # Get all unique nodes
        unique_nodes = np.unique(np.concatenate([sources, targets]))
        
        # Subsampling
        if subsample_ratio < 1.0:
            np.random.seed(seed)
            num_keep = int(len(unique_nodes) * subsample_ratio)
            keep_nodes = np.random.choice(unique_nodes, size=num_keep, replace=False)
            unique_nodes = np.sort(keep_nodes)
            
            # Filter edges to induce subgraph (keep edges only if both endpoints are in subset)
            mask_source = np.isin(sources, unique_nodes)
            mask_target = np.isin(targets, unique_nodes)
            mask_edges = mask_source & mask_target
            
            sources = sources[mask_edges]
            targets = targets[mask_edges]
            
        self.num_nodes = len(unique_nodes)
        
        # Map original IDs to 0..N-1
        node_map = {node: i for i, node in enumerate(unique_nodes)}
        
        # Vectorized mapping is faster for large arrays
        # But dictionary mapping is simple. Let's stick to dictionary for now as it is robust.
        # For very large graphs, we might optimize this mapping.
        mapped_sources = np.array([node_map[s] for s in sources], dtype=np.int32)
        mapped_targets = np.array([node_map[t] for t in targets], dtype=np.int32)
        
        # Create adjacency matrix
        # Weights are 1 for unweighted graph
        data = np.ones(len(mapped_sources), dtype=np.int32)
        
        # Create symmetric matrix
        # Duplicate edges might exist in raw data, or be created by symmetry
        # We want a simple undirected graph, so we treat it as binary.
        
        # Build COO matrix first
        adj = sparse.coo_matrix((data, (mapped_sources, mapped_targets)), 
                                shape=(self.num_nodes, self.num_nodes))
        
        # Symmetrize: strictly speaking, we want (u, v) and (v, u) to be present
        # The sum will add them.
        adj = adj + adj.T
        
        # Convert to CSR for arithmetic efficiency
        adj = adj.tocsr()
        
        # Handle duplicate edges (e.g. if input had u->v and v->u, or multiple ratings)
        # We just want binary adjacency: 1 if edge exists, 0 otherwise.
        adj.data = np.ones_like(adj.data)
        
        # Remove self-loops (optional, but "cut" usually implies simple graphs)
        adj.setdiag(0)
        adj.eliminate_zeros()
        
        self.adj_matrix = adj
        
        # Pre-calculate degrees: D = A * 1
        # sparse matrix * vector of ones gives row sums (degrees)
        self.degrees = np.array(self.adj_matrix.sum(axis=1)).flatten()

    def get_subgraph_edge_count(self, mask):
        """
        Computes the number of edges where both endpoints are in the subset defined by mask.
        
        Args:
            mask: Boolean array of shape (num_nodes,).
            
        Returns:
            int: Number of edges inside the subgraph.
        """
        if len(mask) != self.num_nodes:
            raise ValueError(f"Mask length {len(mask)} does not match number of nodes {self.num_nodes}")
            
        # Efficient calculation: 1/2 * x^T * A * x
        # Since A is binary symmetric, x^T A x counts 2 * edges (once for u->v, once for v->u)
        
        # Simply selecting the submatrix and summing might be intuitive but maybe slower than vector ops?
        # Actually doing (A * mask) . dot (mask) is efficient vector-matrix-vector mult.
        
        # Convert boolean mask to integer/float vector for multiplication
        x = mask.astype(int)
        
        # A * x gives a vector where ith element is number of neighbors of i that are in mask
        neighbors_in_subset = self.adj_matrix.dot(x)
        
        # Dot product with x again sums these up only for nodes i that are themselves in mask
        total_degree_sum = neighbors_in_subset.dot(x)
        
        return total_degree_sum // 2

    def get_cut_size(self, mask):
        """
        Computes the number of edges between the subset defined by mask and its complement.
        
        Args:
            mask: Boolean array of shape (num_nodes,).
            
        Returns:
            int: Cut size.
        """
        if len(mask) != self.num_nodes:
            raise ValueError(f"Mask length {len(mask)} does not match number of nodes {self.num_nodes}")

        # Optimization: Cut(S, S_bar) = Vol(S) - 2 * Edges(S)
        # Vol(S) is sum of degrees of nodes in S
        
        # Sum of degrees for nodes in S
        vol_s = self.degrees[mask].sum()
        
        edges_inside = self.get_subgraph_edge_count(mask)
        
        return vol_s - 2 * edges_inside
