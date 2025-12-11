# Post-Quantum Blockchain Consensus Implementation

This is an implementation of the **Post-Quantum Proof of Work (PoW) Consensus Mechanism** from the paper:

> *"On the Construction of a Post-Quantum Blockchain"* by Chen et al. (IEEE DSC 2021)

## Overview

This implementation focuses **exclusively on Section IV** (the consensus mechanism), leaving out the transaction-related components from Section V.

### The Core Innovation

Instead of Bitcoin's hash-based PoW (find nonce such that SHA256(block) < target), this system uses:

**Mining**: Solve a system of multivariate quadratic equations using Gröbner basis algorithms

This provides two key advantages:
1. **Quantum Resistance**: MQ (Multivariate Quadratic) problem is NP-hard with no known quantum speedup
2. **Memory-Hard**: Gröbner basis computation requires significant memory, resisting ASIC mining

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     CONSENSUS WORKFLOW                          │
│                      (Figure 4 from paper)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Hash of Father Block ──┐                                      │
│                          ├──► SHA256 ──► Seed                   │
│   Nonce ─────────────────┘                  │                   │
│                                            ▼                    │
│                                      PRNG(Seed, i)              │
│                                            │                    │
│                                            ▼                    │
│                              System of Quadratic Equations      │
│                              F = (f₁, f₂, ..., fₙ)              │
│                                            │                    │
│                                            ▼                    │
│                               Gröbner Basis Algorithm           │
│                                     (F4)                        │
│                                            │                    │
│                                            ▼                    │
│                              Solution = (x₁, x₂, ..., xₙ)       │
│                                            │                    │
│                                            ▼                    │
│                            SHA256(Solution) ≤ Target?           │
│                                    │              │             │
│                                   YES             NO            │
│                                    │              │             │
│                                    ▼              ▼             │
│                              Generate Block    Nonce++          │
│                                                   │             │
│                                            (loop back)          │
└─────────────────────────────────────────────────────────────────┘
```

## Files

| File | Lines | Description |
|------|-------|-------------|
| `finite_field.py` | 187 | Finite field arithmetic (F_q) |
| `equation_system.py` | 348 | Generates MQ equation systems from seed |
| `groebner_solver.py` | 563 | Gröbner basis solver for mining |
| `blockchain.py` | 479 | Block structure, mining, verification, DAA |
| `chain.py` | 467 | Chain management and simulation |

## Mathematical Foundation

### The MQ Problem

Each equation has the form (Equation 2 from paper):

```
∑∑ α_ij · x_i · x_j + ∑ β_j · x_j + γ = 0
```

Where:
- `n` variables: x₁, x₂, ..., xₙ
- `m` equations: f₁, f₂, ..., fₘ (typically m = n)
- All coefficients from finite field F_q

### Deterministic Generation

```
Seed = SHA256(SHA256(Block_{i-1}) | Nonce)
Coefficients = PRNG(Seed) mod q
```

This ensures all nodes generate the **same equations** from the same seed.

### Solving via Gröbner Basis

The Gröbner basis transforms the polynomial system into an equivalent system with useful properties (similar to Gaussian elimination for linear systems, but for polynomials).

From **Table II** in the paper (using F4 algorithm on F_32):

| Parameters | Time | Memory |
|------------|------|--------|
| n=6, m=6 | 0.020s | 0.9 MB |
| n=8, m=8 | 0.240s | 32.1 MB |
| n=10, m=10 | 3.609s | 828.2 MB |
| n=12, m=12 | 326.4s | 7,093 MB |
| n=18, m=18 | 11,515s | 170,152 MB |

**Recommended production parameters**: n=12, m=12, q=32

## Algorithms from Paper

### Algorithm 1: Blockchain Network Setup
- Set PoWLimit (maximum target)
- Set DAA (Difficulty Adjustment Algorithm)
- Select Gröbner solver (F4)

### Algorithm 2: Mining (Miner Nodes)
```
while true:
    if SHA256(x₁, x₂, ..., xₙ) ≤ PoWLimit/D_i:
        Broadcast new Block_i
        break
    else:
        Nonce++
        Seed = SHA256(SHA256(Block_{i-1}) | Nonce)
        Random_i = PRNG(Seed, i) / q
        F = construct_equations(Random_i)
        (x₁, ..., xₙ) = F4(f₁, ..., fₙ)
```

### Algorithm 3: Verification (All Nodes)
```
1. Seed = SHA256(SHA256(Block_{i-1}) | Nonce)
2. Random_i = PRNG(Seed, i) / q
3. Reconstruct F = (f₁, ..., fₙ)
4. Substitute Solution into F
5. Verify all equations equal 0
6. Verify SHA256(Solution) ≤ Target
```

## Usage

### Run the Simulation
```bash
cd pq_blockchain
python3 chain.py
```

### Test Individual Components
```bash
python3 finite_field.py      # Test field arithmetic
python3 equation_system.py   # Test equation generation
python3 groebner_solver.py   # Test solver
python3 blockchain.py        # Test mining/verification
```

## Key Properties Demonstrated

1. **Post-Quantum Security**: Uses NP-hard MQ problem
2. **Memory Mining**: Gröbner basis requires memory ∝ system size
3. **Verification Asymmetry**: Verification is ~20,000x faster than mining
4. **Difficulty Adjustment**: DAA maintains stable block times
5. **Tamper Detection**: Modified solutions fail verification

## Configuration

```python
MiningConfig(
    field_size=31,      # q: finite field size (paper uses 32)
    n_vars=4,           # n: number of variables
    n_equations=4,      # m: number of equations (m=n typical)
    pow_limit=2**240,   # PoWLimit
    target_block_time=60,  # seconds per block
    difficulty_adjustment_window=10  # blocks for DAA
)
```

## References

- [Original Paper](https://ieeexplore.ieee.org/document/9346253)
- Faugère, J. (1999). "A new efficient algorithm for computing Gröbner bases (F4)"
- Courtois et al. (2000). "Efficient algorithms for solving overdefined systems of multivariate polynomial equations"

## License

Educational/research implementation.
