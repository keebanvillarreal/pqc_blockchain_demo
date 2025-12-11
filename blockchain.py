"""
Post-Quantum PoW Consensus Implementation

This module implements the consensus mechanism (Section IV of the paper).

The key innovations are:

1. PROOF OF WORK: Instead of finding a hash below a target, miners must:
   a) Generate a system of multivariate quadratic equations from seed
   b) Solve the system using Gröbner basis algorithm (memory + computation)
   c) Check if SHA256(solution) ≤ PoWLimit / Difficulty

2. DIFFICULTY ADJUSTMENT: The paper uses a DAA (Difficulty Adjustment Algorithm)
   to maintain stable block times, similar to Bitcoin but applied to our new PoW.

3. VERIFICATION: Any node can quickly verify a solution by:
   a) Regenerating the equations from seed
   b) Substituting the solution to verify it satisfies all equations
   c) Checking the hash condition

MEMORY MINING PROPERTY (from Table II in paper):
================================================
The Gröbner basis computation requires significant memory, making this
resistant to ASIC mining. The memory usage grows with n (number of variables):
- n=6, m=6, q=32:  0.020s,    0.9 MB
- n=8, m=8, q=32:  0.240s,   32.1 MB  
- n=10, m=10, q=32: 3.609s,  828.2 MB
- n=12, m=12, q=32: 326.4s, 7093 MB (recommended for production)

This is a key advantage over Bitcoin's hash-based PoW.
"""

import hashlib
import struct
import time
from typing import List, Optional, Tuple
from dataclasses import dataclass
from finite_field import FiniteField, FiniteFieldElement
from equation_system import EquationSystemGenerator, MultivariatePolynomial
from groebner_solver import solve_equations


def sha256(data: bytes) -> bytes:
    """Compute SHA-256 hash"""
    return hashlib.sha256(data).digest()


def double_sha256(data: bytes) -> bytes:
    """Compute double SHA-256 (as used in Bitcoin)"""
    return sha256(sha256(data))


def hash_to_int(hash_bytes: bytes) -> int:
    """Convert hash bytes to integer for comparison with target"""
    return int.from_bytes(hash_bytes, 'big')


def int_to_hash(value: int) -> bytes:
    """Convert integer back to 32-byte hash"""
    return value.to_bytes(32, 'big')


@dataclass
class BlockHeader:
    """
    Block header structure for the post-quantum blockchain.
    
    This is a simplified header focused on consensus fields.
    In production, additional fields (merkle root, etc.) would be added.
    """
    version: int                    # Protocol version
    prev_block_hash: bytes          # Hash of previous block (32 bytes)
    timestamp: int                  # Unix timestamp
    difficulty: int                 # Current difficulty level
    nonce: int                      # Nonce used to generate equations
    solution: List[int]             # Solution to the equation system
    pow_limit: int                  # Maximum target value
    data_hash: bytes = None         # Optional: hash of block data/payload
    
    def __post_init__(self):
        if self.data_hash is None:
            self.data_hash = bytes(32)  # Empty data hash
    
    def serialize(self) -> bytes:
        """Serialize header for hashing (excluding solution for seed generation)"""
        data = struct.pack('>I', self.version)           # 4 bytes
        data += self.prev_block_hash                      # 32 bytes
        data += self.data_hash                            # 32 bytes
        data += struct.pack('>Q', self.timestamp)        # 8 bytes
        data += struct.pack('>I', self.difficulty)       # 4 bytes
        data += struct.pack('>Q', self.nonce)            # 8 bytes
        return data
    
    def serialize_with_solution(self) -> bytes:
        """Full serialization including solution"""
        data = self.serialize()
        data += struct.pack('>I', len(self.solution))    # 4 bytes
        for val in self.solution:
            data += struct.pack('>I', val)               # 4 bytes each
        return data
    
    def hash(self) -> bytes:
        """Compute block hash"""
        return double_sha256(self.serialize_with_solution())


@dataclass  
class Block:
    """
    Block structure focused on consensus.
    """
    header: BlockHeader
    
    def hash(self) -> bytes:
        return self.header.hash()
    
    def to_dict(self) -> dict:
        """Convert to dictionary for display/serialization"""
        return {
            'version': self.header.version,
            'prev_block_hash': self.header.prev_block_hash.hex(),
            'timestamp': self.header.timestamp,
            'difficulty': self.header.difficulty,
            'nonce': self.header.nonce,
            'solution': self.header.solution,
            'pow_limit': self.header.pow_limit,
            'block_hash': self.hash().hex()
        }


class MiningConfig:
    """
    Configuration parameters for the PoW consensus.
    
    From Table II in the paper, recommended production parameters:
    - n=12, m=12, q=32 gives ~326s mining time and ~7GB memory
    
    For testing, smaller parameters can be used.
    """
    
    def __init__(self, 
                 field_size: int = 31,      # Size of finite field (q), paper uses 32
                 n_vars: int = 4,           # Number of variables (n)
                 n_equations: int = 4,       # Number of equations (m), typically m=n
                 pow_limit: int = 2**240,   # Maximum target (PoWLimit from paper)
                 target_block_time: int = 60,  # Target seconds per block
                 difficulty_adjustment_window: int = 10  # Blocks for DAA
                 ):
        self.field_size = field_size
        self.n_vars = n_vars
        self.n_equations = n_equations
        self.pow_limit = pow_limit
        self.target_block_time = target_block_time
        self.difficulty_adjustment_window = difficulty_adjustment_window
        
        # Create field and equation generator
        self.field = FiniteField(field_size)
        self.eq_generator = EquationSystemGenerator(
            self.field, n_vars, n_equations
        )


class PostQuantumMiner:
    """
    Miner implementation following Algorithm 2 from the paper.
    
    The mining loop:
    1. Generate seed from previous block hash and nonce
    2. Generate equation system from seed
    3. Solve equation system using Gröbner basis
    4. Check if SHA256(solution) ≤ target
    5. If not, increment nonce and repeat
    """
    
    def __init__(self, config: MiningConfig):
        self.config = config
    
    def compute_target(self, difficulty: int) -> int:
        """
        Compute target value from difficulty.
        
        Target = PoWLimit / Difficulty
        Lower target = harder to find valid solution
        """
        if difficulty <= 0:
            difficulty = 1
        return self.config.pow_limit // difficulty
    
    def generate_equations(self, prev_hash: bytes, nonce: int) -> Tuple[bytes, List[MultivariatePolynomial]]:
        """
        Generate the equation system for mining.
        
        Returns:
            (seed, equations) tuple
        """
        seed = self.config.eq_generator.generate_seed(prev_hash, nonce)
        equations = self.config.eq_generator.generate_system(seed)
        return seed, equations
    
    def hash_solution(self, solution: List[FiniteFieldElement]) -> bytes:
        """
        Hash the solution for target comparison.
        
        SHA256(x_1, x_2, ..., x_n)
        """
        data = b''
        for val in solution:
            data += struct.pack('>I', val.value)
        return sha256(data)
    
    def check_solution_meets_target(self, solution: List[FiniteFieldElement], 
                                    target: int) -> bool:
        """
        Check if SHA256(solution) ≤ target.
        """
        solution_hash = self.hash_solution(solution)
        solution_int = hash_to_int(solution_hash)
        return solution_int <= target
    
    def mine_block(self, prev_hash: bytes,
                   difficulty: int, max_nonces: int = 1000,
                   verbose: bool = False) -> Optional[BlockHeader]:
        """
        Attempt to mine a block - Algorithm 2 from the paper.
        
        The mining loop:
        1. For each nonce:
           a. Generate seed = SHA256(SHA256(Block_{i-1}) | Nonce)
           b. Generate equation system F from seed using PRNG
           c. Solve (x_1,...,x_n) = GrobnerBasis(f_1,...,f_n)
           d. Check if SHA256(x_1,...,x_n) ≤ PoWLimit/Difficulty
        
        Args:
            prev_hash: Hash of previous block (SHA256(Block_{i-1}))
            difficulty: Current difficulty (D_i in paper)
            max_nonces: Maximum nonces to try
            verbose: Print progress
        
        Returns:
            BlockHeader if successful, None otherwise
        """
        target = self.compute_target(difficulty)
        timestamp = int(time.time())
        
        if verbose:
            print(f"Mining with target: {target:064x}")
            print(f"Parameters: n={self.config.n_vars}, m={self.config.n_equations}, q={self.config.field_size}")
        
        for nonce in range(max_nonces):
            if verbose and nonce % 10 == 0:
                print(f"  Trying nonce {nonce}...")
            
            # Step 1: Generate equations
            seed, equations = self.generate_equations(prev_hash, nonce)
            
            # Step 2: Solve equations (the expensive part!)
            try:
                solutions = solve_equations(equations, verbose=False)
            except Exception as e:
                if verbose:
                    print(f"  Solver error at nonce {nonce}: {e}")
                continue
            
            if not solutions:
                # No solution found - this can happen for random systems
                continue
            
            # Step 3: Check each solution against target
            for sol in solutions:
                if self.check_solution_meets_target(sol, target):
                    # Found valid solution!
                    if verbose:
                        print(f"  FOUND VALID SOLUTION at nonce {nonce}!")
                    
                    header = BlockHeader(
                        version=1,
                        prev_block_hash=prev_hash,
                        timestamp=timestamp,
                        difficulty=difficulty,
                        nonce=nonce,
                        solution=[v.value for v in sol],
                        pow_limit=self.config.pow_limit
                    )
                    return header
        
        if verbose:
            print(f"No valid solution found in {max_nonces} nonces")
        return None


class BlockValidator:
    """
    Validates blocks according to the paper's verification algorithm (Algorithm 3).
    
    Verification is much faster than mining:
    1. Regenerate equations from seed
    2. Substitute solution and check all equations equal zero
    3. Check hash condition
    """
    
    def __init__(self, config: MiningConfig):
        self.config = config
    
    def verify_solution(self, header: BlockHeader) -> Tuple[bool, str]:
        """
        Verify a block's proof of work.
        
        This is Algorithm 3 from the paper.
        
        Returns:
            (is_valid, message) tuple
        """
        # Step 1: Regenerate equations from seed
        seed = self.config.eq_generator.generate_seed(
            header.prev_block_hash, header.nonce
        )
        equations = self.config.eq_generator.generate_system(seed)
        
        # Step 2: Convert solution to field elements
        if len(header.solution) != self.config.n_vars:
            return False, f"Invalid solution length: expected {self.config.n_vars}, got {len(header.solution)}"
        
        solution = [
            self.config.field.element(v) 
            for v in header.solution
        ]
        
        # Step 3: Verify solution satisfies all equations
        for i, eq in enumerate(equations):
            result = eq.evaluate(solution)
            if not result.is_zero():
                return False, f"Equation {i} not satisfied: f({header.solution}) = {result.value}"
        
        # Step 4: Check hash condition
        miner = PostQuantumMiner(self.config)
        target = miner.compute_target(header.difficulty)
        solution_hash = miner.hash_solution(solution)
        solution_int = hash_to_int(solution_hash)
        
        if solution_int > target:
            return False, f"Hash {solution_int:x} exceeds target {target:x}"
        
        return True, "Valid block"


class DifficultyAdjuster:
    """
    Dynamic Difficulty Adjustment (DAA) as mentioned in Section IV of the paper.
    
    From the paper: "We add dynamic difficulty adjustment (DAA) algorithms 
    in our consensus construction."
    
    The target value is computed as: Target = PoWLimit / D_i
    where D_i is adjusted based on recent block times.
    
    This ensures stable block generation rate regardless of network hashrate.
    """
    
    def __init__(self, config: MiningConfig):
        self.config = config
    
    def calculate_new_difficulty(self, 
                                 current_difficulty: int,
                                 block_timestamps: List[int]) -> int:
        """
        Calculate new difficulty based on recent block times.
        
        Args:
            current_difficulty: Current difficulty level
            block_timestamps: Timestamps of recent blocks (oldest to newest)
        
        Returns:
            New difficulty value
        """
        if len(block_timestamps) < 2:
            return current_difficulty
        
        # Calculate actual time for recent blocks
        actual_time = block_timestamps[-1] - block_timestamps[0]
        expected_time = (len(block_timestamps) - 1) * self.config.target_block_time
        
        if actual_time <= 0:
            actual_time = 1
        
        # Adjust difficulty proportionally
        # If blocks are too fast, increase difficulty
        # If blocks are too slow, decrease difficulty
        ratio = expected_time / actual_time
        
        # Clamp ratio to prevent extreme adjustments (max 4x change)
        ratio = max(0.25, min(4.0, ratio))
        
        new_difficulty = int(current_difficulty * ratio)
        
        # Ensure minimum difficulty of 1
        return max(1, new_difficulty)


# Test the consensus mechanism
if __name__ == "__main__":
    print("Post-Quantum PoW Consensus - Mining Test")
    print("=" * 70)
    
    # Use small parameters for quick testing
    # Paper recommends n=12, m=12, q=32 for production
    config = MiningConfig(
        field_size=7,       # Small field for testing
        n_vars=3,           # 3 variables  
        n_equations=3,      # 3 equations
        pow_limit=2**250,   # Easy target for testing
        target_block_time=10
    )
    
    print(f"Configuration (Section IV parameters):")
    print(f"  Finite Field: F_{config.field_size}")
    print(f"  Variables (n): {config.n_vars}")
    print(f"  Equations (m): {config.n_equations}")
    print(f"  PoW Limit: 2^{config.pow_limit.bit_length()-1}")
    
    # Create genesis block hash (all zeros for first block)
    genesis_hash = bytes(32)
    
    print("\n" + "-" * 70)
    print("Mining Block 1 (Algorithm 2 from paper)...")
    print("-" * 70)
    
    miner = PostQuantumMiner(config)
    validator = BlockValidator(config)
    
    start_time = time.time()
    header = miner.mine_block(
        genesis_hash, 
        difficulty=1,  # Start with easy difficulty
        max_nonces=100,
        verbose=True
    )
    mining_time = time.time() - start_time
    
    if header:
        print(f"\nBlock mined in {mining_time:.2f} seconds!")
        print(f"  Nonce: {header.nonce}")
        print(f"  Solution: {header.solution}")
        print(f"  Block hash: {header.hash().hex()[:32]}...")
        
        # Verify the block (Algorithm 3 from paper)
        print("\nVerifying block (Algorithm 3)...")
        is_valid, message = validator.verify_solution(header)
        print(f"  Verification: {message}")
        
        # Create full block
        block = Block(header=header)
        print(f"\nBlock details:")
        for key, value in block.to_dict().items():
            print(f"  {key}: {value}")
    else:
        print("\nFailed to mine block")
    
    # Test difficulty adjustment
    print("\n" + "-" * 70)
    print("Testing Difficulty Adjustment (DAA)")
    print("-" * 70)
    
    adjuster = DifficultyAdjuster(config)
    
    # Simulate blocks that are too fast (5 seconds each instead of 10)
    timestamps_fast = [0, 5, 10, 15, 20]
    new_diff = adjuster.calculate_new_difficulty(100, timestamps_fast)
    print(f"Fast blocks (5s avg): difficulty {100} -> {new_diff} (increased)")
    
    # Simulate blocks that are too slow (20 seconds each)
    timestamps_slow = [0, 20, 40, 60, 80]
    new_diff = adjuster.calculate_new_difficulty(100, timestamps_slow)
    print(f"Slow blocks (20s avg): difficulty {100} -> {new_diff} (decreased)")
    
    # Simulate blocks at target rate
    timestamps_target = [0, 10, 20, 30, 40]
    new_diff = adjuster.calculate_new_difficulty(100, timestamps_target)
    print(f"Target blocks (10s avg): difficulty {100} -> {new_diff} (unchanged)")
