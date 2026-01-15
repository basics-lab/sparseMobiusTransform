# SMT vs ASMT Algorithm Comparison

This document provides a comprehensive comparison of the **Sparse Mobius Transform (SMT)** and **Adaptive Sparse Mobius Transform (ASMT)** algorithms implemented in this repository.

## Table of Contents

1. [Overview](#overview)
2. [SMT Algorithm](#smt-algorithm)
3. [ASMT Algorithm](#asmt-algorithm)
4. [ASMT vs FASMT: Detailed Comparison](#asmt-vs-fasmt-detailed-comparison)
5. [SMT vs ASMT Comparison](#smt-vs-asmt-comparison)
6. [Implementation Details](#implementation-details)
7. [Usage Guidelines](#usage-guidelines)

---

## Overview

Both algorithms compute the **sparse Boolean Mobius transform** of a signal, recovering nonzero coefficients from a sparse representation. The key difference lies in their query strategies:

- **SMT**: Uses **fixed batch subsampling** with pre-determined matrices
- **ASMT**: Uses **adaptive binary search** with on-demand measurements

### Problem Setting

Given a signal `f: {0,1}^n -> R` with a sparse Mobius transform (at most `k` nonzero coefficients), both algorithms aim to recover the locations and values of these coefficients using as few samples as possible.

---

## SMT Algorithm

**File**: `smt/smt.py`

### Mathematical Formulation

SMT uses a **peeling decoder** that operates on aliased bins created by subsampling matrices.

#### Key Components

1. **Subsampling Matrices (M)**: Binary matrices of size `n x b` that create affine subspaces
   - Each matrix M hashes indices into `2^b` bins
   - Index `k` maps to bin `j` where `j = M^T @ k (mod 2)`

2. **Delay Matrices (D)**: Binary matrices for channel/source coding
   - Enable singleton detection via signature matching
   - Signature: `s = 1 - (D @ k > 0)`

3. **Walsh-Hadamard Transform (WHT)**: Applied to subsampled measurements
   - Transforms time-domain samples to frequency-domain bins
   - Each bin contains aliased coefficients

#### Algorithm Steps

```
1. INITIALIZATION
   - Generate M matrices (subsampling) and D matrices (delays)
   - Compute WHT on all subsampled measurements -> U matrices

2. PEELING LOOP (max 1000 iterations)
   For each bin (i, j):
     a) Check if bin energy exceeds noise threshold
     b) Attempt singleton detection:
        - Decode index k from bin column
        - Compute signature: s = 1 - (D @ k > 0)
        - Estimate amplitude: rho = (s . col) / sum(s)
        - Verify: residual = col - rho * s
     c) Classify bin:
        - ZEROTON: energy < cutoff (empty)
        - SINGLETON: residual < cutoff and bin matches
        - MULTITON: otherwise (multiple coefficients)

3. PEEL SINGLETONS
   - Remove detected singletons from all affected bins
   - Update residual measurements
   - Re-check affected bins

4. OUTPUT
   - Dictionary mapping indices to coefficient values
```

#### Key Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `b` | Subsampling dimension (creates 2^b bins) | 4-8 |
| `num_subsample` | Number of M matrices | 2-5 |
| `num_repeat` | Number of delay repetitions | 1-3 |
| `noise_sd` | Noise standard deviation for thresholding | 0-1 |

#### Singleton Detection Methods

Located in `smt/reconstruct.py`:

1. **Identity**: Direct observation (noiseless case)
2. **NSO (Noise-aware)**: Uses repetition codes with random offsets
3. **Coded**: External decoder for error-correcting codes
4. **MLE**: Maximum likelihood (exhaustive search)

#### Noise Threshold

```python
gamma = 0.5
cutoff = 1e-9 + (1 + gamma) * (noise_sd ** 2)
```

---

## ASMT Algorithm

The ASMT family uses **adaptive binary search** to identify nonzero coefficients. Two implementations exist:

- **ASMT** (`asmt/asmt.py`): Triangular matrix approach with level-by-level processing
- **FASMT** (`asmt/fasmt.py`): Fully adaptive tree-based approach with online peeling

---

## ASMT vs FASMT: Detailed Comparison

This section provides an in-depth comparison of the two ASMT variants.

### ASMT Baseline (`asmt/asmt.py`)

**Core Mechanism**: Level-by-level processing with triangular matrix solving

#### Data Structures

```python
# Key variables (lines 48-53)
M = np.ones((1, 1), dtype=np.int32)       # Growing lower triangular matrix
loc = np.zeros((1, 0), dtype=np.int32)    # Partial binary locations (grows column-by-column)
val = measurements[0] * np.ones(1)         # Current coefficient values
H = get_Ms(n, b, ...)                      # Subsampling matrix (generated once)
```

#### Algorithm Flow

**File**: `asmt/asmt.py`, lines 58-88

```python
for i in range(b):  # Level-by-level iteration
    if len(val) == 0:
        break
    b1 = loc.shape[1]

    # Step 1: Compute measurement positions using H matrix
    measurement_positions = np.array(((1 - loc) @ H[:, :b1].T + H[:, b1]) < 0.5, dtype=np.int32)

    # Step 2: Take measurements for ALL active nodes at this level
    measurements_new = signal.subsample(measurement_positions)

    # Step 3: Solve triangular system to split coefficients
    coefs_left = la.solve_triangular(M, measurements_new, lower=True)
    coefs_right = val - coefs_left

    # Step 4: Keep only nonzero branches
    support_first = np.where(np.abs(coefs_left) > eps)[0]
    support_second = np.where(np.abs(coefs_right) > eps)[0]

    # Step 5: Update M matrix (grows as tree expands)
    M_prev = M.copy()
    M = np.zeros((dim1 + dim2, dim1 + dim2), dtype=np.int32)
    M[:dim1, :dim1] = M_prev[support_first][:, support_first]
    M[dim1:, :dim1] = M_prev[support_second][:, support_first]
    M[dim1:, dim1:] = M_prev[support_second][:, support_second]

    # Step 6: Update loc (add column for new level)
    locs_first[:, :b1] = loc[support_first]
    locs_second[:, :b1] = loc[support_second]
    locs_second[:, -1] = 1  # Mark right branch
```

**Decoding**: External decoder called at END (line 90-92)
```python
for ix, (l, m) in enumerate(zip(loc, val)):
    k_dec, success = decoder(H.T, l[np.newaxis, :].astype(bool).T)
    transform[tuple(k_dec)] = m
```

#### Key Characteristics

- **Batch Processing**: All active nodes at each level processed together
- **Fixed Iterations**: Exactly `b` levels (for loop)
- **Matrix-Based**: Uses `scipy.linalg.solve_triangular` for coefficient splitting
- **Deferred Decoding**: Index recovery happens after all levels complete
- **No Online Peeling**: Coefficients not subtracted during computation

---

### FASMT (`asmt/fasmt.py`)

**Core Mechanism**: Greedy node-by-node processing with online peeling

#### Data Structures

```python
# Key variables (lines 26-35)
val_tree = SortedDict({tuple(): {"value": ..., "nz": [], "stage_depth": 0}})  # Tree of nodes
h_tree = {}                                    # Partition masks for each node
loc_peeled = np.zeros((0, n), dtype=np.int32)  # Found coefficient locations
val_peeled = np.zeros(0, dtype=np.int32)       # Found coefficient values
```

#### Peeling Function

**File**: `asmt/fasmt.py`, lines 28-29

```python
def sample_peeled_function(positions_batch):
    # Subtract already-found coefficients from new measurements
    return (((1 - positions_batch) @ loc_peeled.T) < 0.5) @ val_peeled
```

#### Algorithm Flow

**File**: `asmt/fasmt.py`, lines 43-95

```python
while len(val_tree) > 0:
    # Step 1: Pop leftmost node (greedy ordering via SortedDict)
    loc, data = val_tree.popitem(index=0)
    val = data["value"]
    nz = data["nz"]
    stage_depth = data["stage_depth"]

    # Step 2: Compute search space (variables still to determine)
    if len(loc) > 0:
        H = np.stack([h_tree[tuple(loc[:i])] for i in range(len(loc))], axis=0)
        if stage_depth == 0:
            search_loc = [i for i in range(n) if i not in nz]
        else:
            H_stage = H[-stage_depth:]
            loc_stage = loc[-stage_depth:]
            search_loc = np.where(np.prod((H_stage.T + (1 - np.array(loc_stage))) % 2, axis=1))[0]
    else:
        search_loc = np.arange(n)

    # Step 3: Create random partition mask
    h_new = np.zeros(n)
    mask_loc = np.random.choice(search_loc, size=(len(search_loc) + 1) // 2, replace=False)
    h_new[mask_loc] = 1
    h_tree[loc] = h_new

    # Step 4: Take SINGLE measurement (minus peeled coefficients)
    measurement_position = np.array(((1 - np.array(loc)) @ H + h_new) < 0.5, dtype=np.int32)
    measurement_new = signal.subsample(measurement_position[np.newaxis, :]) - \
                      sample_peeled_function(measurement_position[np.newaxis, :])

    # Step 5: Process result
    if abs(measurement_new) > eps:
        # Push left child (measurement is nonzero)
        val_tree[loc + (0,)] = {"value": measurement_new, "nz": nz, "stage_depth": stage_depth + 1}

    if abs(val - measurement_new) > eps:
        if len(search_loc) == 1:
            # ISOLATED: Only one variable left - PEEL immediately
            new_nz = nz + [search_loc[0]]
            loc_new = np.zeros(n, dtype=np.int32)
            loc_new[new_nz] = 1

            # Take isolation measurement and peel
            measurement_new = signal.subsample(loc_new[np.newaxis, :]) - \
                              sample_peeled_function(loc_new[np.newaxis, :])
            if np.abs(measurement_new) > eps:
                loc_peeled = np.concatenate([loc_peeled, loc_new[np.newaxis, :]], axis=0)
                val_peeled = np.append(val_peeled, [measurement_new])
                transform[tuple(loc_new)] = measurement_new

            # Continue with residual
            val_tree[loc + (1,)] = {"value": val - measurement_new, "nz": new_nz, "stage_depth": 0}
        else:
            # Push right child (continue search)
            val_tree[loc + (1,)] = {"value": val - measurement_new, "nz": nz, "stage_depth": stage_depth + 1}
```

#### Key Characteristics

- **Sequential Processing**: One node at a time (greedy via SortedDict)
- **Dynamic Iterations**: While loop until tree empty
- **Tree-Based**: Uses dictionaries (`val_tree`, `h_tree`) for state
- **Online Peeling**: Coefficients subtracted immediately when found
- **Direct Recovery**: No external decoder needed

---

### Side-by-Side Comparison

| Aspect | ASMT (`asmt.py`) | FASMT (`fasmt.py`) |
|--------|------------------|-------------------|
| **Processing Order** | Level-by-level (breadth-first) | Greedy (leftmost node first) |
| **Samples per Iteration** | Multiple (all nodes at level) | Single (one node) |
| **Loop Structure** | `for i in range(b)` | `while len(val_tree) > 0` |
| **Coefficient Recovery** | External decoder at end | Online peeling |
| **Matrix Structure** | Explicit triangular M | Implicit via h_tree masks |
| **Search Space Tracking** | Via triangular solve | Explicit `search_loc` computation |
| **Dependencies** | `scipy.linalg` | `sortedcontainers.SortedDict` |
| **Memory Pattern** | Growing triangular matrix | Tree node dictionaries |

### Measurement Position Computation

**ASMT** (line 62):
```python
measurement_positions = np.array(((1 - loc) @ H[:, :b1].T + H[:, b1]) < 0.5, dtype=np.int32)
```

**FASMT** (line 63):
```python
measurement_position = np.array(((1 - np.array(loc)) @ H + h_new) < 0.5, dtype=np.int32)
```

### Coefficient Splitting

**ASMT** - Uses linear algebra (lines 70-71):
```python
coefs_left = la.solve_triangular(M, measurements_new, lower=True)
coefs_right = val - coefs_left
```

**FASMT** - Direct subtraction (lines 76-78, 92-94):
```python
# Left child gets measurement value
val_tree[loc + (0,)] = {"value": measurement_new, ...}
# Right child gets residual
val_tree[loc + (1,)] = {"value": val - measurement_new, ...}
```

### When to Use Which

| Scenario | Recommended |
|----------|-------------|
| **Batch oracle queries available** | ASMT (processes multiple at once) |
| **Sequential query constraint** | FASMT (one query per iteration) |
| **Need intermediate results** | FASMT (online peeling) |
| **Simple implementation** | ASMT (linear algebra approach) |
| **Unknown termination depth** | FASMT (dynamic tree) |

---

## SMT vs ASMT Comparison

### Algorithm Characteristics

| Aspect | SMT | ASMT/FASMT |
|--------|-----|------------|
| **Query Strategy** | Fixed batch subsampling | Adaptive binary search |
| **Measurement Pattern** | All upfront (structured) | On-demand (adaptive) |
| **Peeling Style** | Linear algebra singleton detection | Direct binary tree isolation |
| **Sample Efficiency** | Fixed sample count | Adapts to signal structure |
| **Memory Usage** | Stores full U matrices | Stores partial tree state |

### Computational Complexity

| Operation | SMT | ASMT/FASMT |
|-----------|-----|------------|
| **Sample Complexity** | O(k * b * num_subsample * num_repeat * 2^b) | O(k * log(n)) adaptive |
| **Time Complexity** | O(samples + k * peeling_iterations) | O(k * log(n)) |
| **Space Complexity** | O(num_subsample * 2^b * num_repeat) | O(k + tree_nodes) |

### When to Use Each Algorithm

#### Use SMT When:
- Signal has **moderate sparsity** (k is not too small)
- **Noise is present** (SMT has explicit noise handling)
- **Parallelization** across stages is beneficial
- Fixed sample budget is acceptable

#### Use ASMT/FASMT When:
- Signal is **very sparse** (small k)
- **Sample efficiency** is critical
- Signal structure allows adaptive refinement
- **Noiseless or low-noise** regime

### Noise Handling

| Algorithm | Noise Handling |
|-----------|----------------|
| **SMT** | Explicit noise threshold (`noise_sd` parameter), residual-based singleton verification |
| **ASMT/FASMT** | Implicit via epsilon threshold (`eps = 1e-5`), no explicit noise model |

---

## Implementation Details

### File Structure

```
sparse_transform/
├── __init__.py              # Package exports
├── wrapper.py               # Unified interface (run_transforms)
├── noise_estimator.py       # Automatic noise parameter selection
│
├── smt/                     # SMT Implementation
│   ├── smt.py               # Main algorithm (236 lines)
│   ├── input_signal.py      # Base Signal class
│   ├── input_signal_subsampled.py  # SubsampledSignal class
│   ├── query.py             # Matrix generation (get_Ms, get_D)
│   ├── reconstruct.py       # Singleton detection methods
│   ├── random_group_testing.py  # Group testing utilities
│   └── utils.py             # Shared utilities (fmt, ifmt, etc.)
│
└── asmt/                    # ASMT Implementation
    ├── asmt.py              # Triangular matrix approach (120 lines)
    └── fasmt.py             # Tree-based approach (122 lines)
```

### Shared Utilities

Both algorithms share utilities from `smt/utils.py`:

- **`fmt(x)`**: Fast Mobius Transform (O(N log N))
- **`ifmt(x)`**: Inverse Fast Mobius Transform
- **`bin_to_dec(x)`**: Binary vector to decimal
- **`dec_to_bin_vec(x, n)`**: Decimal to binary vector
- **`calc_hamming_weight(x)`**: Count nonzero bits
- **`binary_ints(m)`**: Generate all m-bit binary vectors

### Configuration via Wrapper

```python
from sparse_transform import run_transforms

# SMT Configuration
smt_config = {
    "n": 1000,
    "b": 6,
    "num_subsample": 3,
    "num_repeat": 1,
    "noise_sd": 0.1,
    "query_method": "group_testing",
    "reconstruct_method": "nso"
}

# ASMT Configuration
asmt_config = {
    "n": 1000,
    "b": 6,
    "t": 10,
    "query_method": "group_testing"
}

result_smt = run_transforms(test_function, "SMT", **smt_config)
result_asmt = run_transforms(test_function, "ASMT", **asmt_config)
```

### Output Format

All algorithms return a dictionary with:

```python
{
    "transform": dict,           # {index_tuple: coefficient_value}
    "runtime": float,            # Total execution time
    "n_samples": int,            # Number of oracle queries
    "locations": list,           # List of nonzero index tuples
    "avg_hamming_weight": float, # Mean Hamming weight of indices
    "max_hamming_weight": int    # Maximum Hamming weight
}
```

---

## Usage Guidelines

### Quick Start

```python
import numpy as np
from sparse_transform import run_transforms

# Define your oracle function
def oracle(query_batch):
    # query_batch: (batch_size, n) binary array
    # Returns: (batch_size,) array of function values
    return your_function_evaluator(query_batch)

# Run SMT
result = run_transforms(oracle, "SMT", n=100, b=5, num_subsample=3,
                        num_repeat=1, noise_sd=0)

# Run ASMT
result = run_transforms(oracle, "ASMT", n=100, b=5, t=10,
                        query_method="group_testing")
```

### Parameter Tuning

#### SMT Parameters
- **b**: Larger b -> more bins -> better separation but more samples
- **num_subsample**: More subsampling groups -> better peeling success
- **noise_sd**: Set based on expected noise level; use `noise_estimator` if unknown

#### ASMT/FASMT Parameters
- **b**: Initial tree branching factor
- **t**: Sparsity parameter for group testing matrices

### Performance Tips

1. **For noiseless signals**: ASMT/FASMT is typically more sample-efficient
2. **For noisy signals**: SMT with proper `noise_sd` tuning is more robust
3. **For very sparse signals (k << n)**: ASMT's adaptive nature excels
4. **For batch query support**: ASMT (level-by-level) may be more efficient
5. **For sequential queries**: FASMT (node-by-node) is natural fit

---

## References

### Key Files

| Component | SMT | ASMT | FASMT |
|-----------|-----|------|-------|
| Main Algorithm | `smt/smt.py:12-236` | `asmt/asmt.py:16-120` | `asmt/fasmt.py:13-122` |
| Coefficient Split | `smt/smt.py:102-130` | `asmt/asmt.py:70-71` | `asmt/fasmt.py:76-94` |
| Matrix Generation | `smt/query.py` | `smt/query.py` (shared) | N/A (dynamic) |
| Transform Utilities | `smt/utils.py:38-63` | `smt/utils.py` (shared) | `smt/utils.py` (shared) |
| Noise Estimation | `noise_estimator.py` | N/A | N/A |

### Algorithm Entry Points

- **SMT**: `sparse_transform.smt.smt.transform()`
- **ASMT**: `sparse_transform.asmt.asmt.transform()`
- **FASMT**: `sparse_transform.asmt.fasmt.transform()`
- **Unified Interface**: `sparse_transform.wrapper.run_transforms()`
