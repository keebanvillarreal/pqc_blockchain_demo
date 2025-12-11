"""
Post-Quantum Blockchain Consensus - Chain Management

This module implements the blockchain chain management focused ONLY on
the consensus mechanism from Section IV of the paper:
- Block generation via PoW
- Chain validation
- Difficulty adjustment

The consensus follows the paper's Algorithms 1, 2, and 3:
- Algorithm 1: Blockchain Network's Running Algorithm
- Algorithm 2: Mining Algorithm of the Miner Nodes  
- Algorithm 3: Verification Algorithm of the Miner Nodes
"""

import time
from typing import List, Optional, Tuple

from finite_field import FiniteField
from blockchain import (
    BlockHeader, Block,
    MiningConfig, PostQuantumMiner, BlockValidator, DifficultyAdjuster,
    sha256
)


class Blockchain:
    """
    Blockchain implementation focused on consensus mechanism.
    
    Implements the workflow from Figure 4 of the paper:
    1. Hash of Father Block + Nonce -> Generate System of Equations
    2. Gröbner Algorithm -> Solution
    3. SHA256(Solution) <= Target? -> Generate Block or Nonce++
    """
    
    def __init__(self, config: MiningConfig):
        self.config = config
        self.chain: List[Block] = []
        self.miner = PostQuantumMiner(config)
        self.validator = BlockValidator(config)
        self.adjuster = DifficultyAdjuster(config)
        
        # Statistics for analysis
        self.total_mining_time = 0.0
        self.total_nonces_tried = 0
        
        # Create genesis block
        self._create_genesis()
    
    def _create_genesis(self):
        """Create the genesis (first) block with trivial PoW"""
        genesis_header = BlockHeader(
            version=1,
            prev_block_hash=bytes(32),  # All zeros for genesis
            timestamp=int(time.time()),
            difficulty=1,
            nonce=0,
            solution=[0] * self.config.n_vars,  # Trivial solution
            pow_limit=self.config.pow_limit
        )
        
        genesis_block = Block(header=genesis_header)
        self.chain.append(genesis_block)
    
    def get_latest_block(self) -> Block:
        """Get the most recent block in the chain"""
        return self.chain[-1]
    
    def get_current_difficulty(self) -> int:
        """
        Calculate current difficulty using DAA.
        
        From paper: Target = PoWLimit / D_i
        where D_i is adjusted based on actual vs expected block times.
        """
        if len(self.chain) < self.config.difficulty_adjustment_window:
            return self.chain[-1].header.difficulty
        
        # Get timestamps from recent blocks for DAA
        recent_blocks = self.chain[-self.config.difficulty_adjustment_window:]
        timestamps = [b.header.timestamp for b in recent_blocks]
        
        current_diff = self.chain[-1].header.difficulty
        return self.adjuster.calculate_new_difficulty(current_diff, timestamps)
    
    def mine_next_block(self, max_nonces: int = 1000,
                        verbose: bool = False) -> Optional[Block]:
        """
        Mine the next block using Algorithm 2.
        
        Steps from paper:
        1. Nonce = Nonce + 1
        2. Seed = SHA256(SHA256(Block_{i-1}) | Nonce)
        3. Random_i = PRNG(Seed, i) / q
        4. Construct system F = (f_1, ..., f_n)
        5. Solve (x_1,...,x_n) = F4(f_1,...,f_n)
        6. If SHA256(x_1,...,x_n) <= PoWLimit/D_i: broadcast block
        
        Args:
            max_nonces: Maximum nonces to try before giving up
            verbose: Print progress information
        
        Returns:
            New Block if mining successful, None otherwise
        """
        prev_block = self.get_latest_block()
        prev_hash = prev_block.hash()
        difficulty = self.get_current_difficulty()
        
        if verbose:
            print(f"\nMining block {len(self.chain)}...")
            print(f"  Previous hash: {prev_hash.hex()[:16]}...")
            print(f"  Difficulty: {difficulty}")
        
        start_time = time.time()
        
        header = self.miner.mine_block(
            prev_hash,
            difficulty,
            max_nonces,
            verbose=verbose
        )
        
        mining_time = time.time() - start_time
        
        if header is None:
            if verbose:
                print(f"  Mining failed after {mining_time:.2f}s")
            return None
        
        # Create block
        block = Block(header=header)
        
        # Update statistics
        self.total_mining_time += mining_time
        self.total_nonces_tried += header.nonce + 1
        
        if verbose:
            print(f"  Mined in {mining_time:.2f}s, nonce={header.nonce}")
        
        return block
    
    def add_block(self, block: Block) -> Tuple[bool, str]:
        """
        Add a block to the chain after verification (Algorithm 3).
        
        Verification steps from paper:
        1. Generate seed from previous block hash and Nonce
        2. Generate system of equations from seed
        3. Substitute Solution into system to verify equations
        4. Check if verification passes
        
        Returns:
            (success, message) tuple
        """
        # Verify proof of work using Algorithm 3
        is_valid, message = self.validator.verify_solution(block.header)
        if not is_valid:
            return False, f"Invalid PoW: {message}"
        
        # Verify chain linkage
        expected_prev_hash = self.get_latest_block().hash()
        if block.header.prev_block_hash != expected_prev_hash:
            return False, "Invalid previous hash - chain broken"
        
        # Verify timestamp ordering
        if block.header.timestamp < self.get_latest_block().header.timestamp:
            return False, "Timestamp before previous block"
        
        # Add to chain
        self.chain.append(block)
        return True, "Block added successfully"
    
    def validate_chain(self) -> Tuple[bool, str]:
        """
        Validate the entire blockchain.
        
        Checks:
        1. Hash linkage between consecutive blocks
        2. PoW validity for each block (Algorithm 3)
        3. Timestamp ordering
        
        Returns:
            (is_valid, message) tuple
        """
        for i in range(1, len(self.chain)):
            block = self.chain[i]
            prev_block = self.chain[i-1]
            
            # Check hash linkage
            if block.header.prev_block_hash != prev_block.hash():
                return False, f"Block {i}: Broken hash chain"
            
            # Check PoW (skip genesis which has trivial solution)
            is_valid, msg = self.validator.verify_solution(block.header)
            if not is_valid:
                return False, f"Block {i}: Invalid PoW - {msg}"
            
            # Check timestamps
            if block.header.timestamp < prev_block.header.timestamp:
                return False, f"Block {i}: Invalid timestamp ordering"
        
        return True, f"Chain valid ({len(self.chain)} blocks)"
    
    def get_statistics(self) -> dict:
        """Get consensus statistics"""
        if len(self.chain) <= 1:
            return {
                'chain_length': len(self.chain),
                'total_mining_time': 0,
                'avg_mining_time': 0,
                'avg_nonces_per_block': 0,
                'current_difficulty': 1
            }
        
        # Calculate block time statistics
        block_times = []
        for i in range(1, len(self.chain)):
            dt = self.chain[i].header.timestamp - self.chain[i-1].header.timestamp
            block_times.append(dt)
        
        avg_block_time = sum(block_times) / len(block_times) if block_times else 0
        mined_blocks = len(self.chain) - 1  # Exclude genesis
        
        return {
            'chain_length': len(self.chain),
            'mined_blocks': mined_blocks,
            'total_mining_time': round(self.total_mining_time, 2),
            'avg_mining_time': round(self.total_mining_time / mined_blocks, 2) if mined_blocks > 0 else 0,
            'avg_block_time': round(avg_block_time, 2),
            'avg_nonces_per_block': round(self.total_nonces_tried / mined_blocks, 1) if mined_blocks > 0 else 0,
            'current_difficulty': self.get_current_difficulty()
        }
    
    def print_chain(self):
        """Display the blockchain"""
        print("\n" + "=" * 70)
        print("BLOCKCHAIN STATE")
        print("=" * 70)
        
        for i, block in enumerate(self.chain):
            block_type = "GENESIS" if i == 0 else f"BLOCK {i}"
            print(f"\n[{block_type}]")
            print(f"  Hash:       {block.hash().hex()[:32]}...")
            print(f"  Prev Hash:  {block.header.prev_block_hash.hex()[:32]}...")
            print(f"  Timestamp:  {time.ctime(block.header.timestamp)}")
            print(f"  Difficulty: {block.header.difficulty}")
            print(f"  Nonce:      {block.header.nonce}")
            print(f"  Solution:   {block.header.solution}")


def run_consensus_simulation(n_blocks: int = 5, verbose: bool = True):
    """
    Run a complete consensus simulation.
    
    Demonstrates the full PoW consensus from Section IV:
    1. Genesis block creation
    2. Mining multiple blocks (Algorithm 2)
    3. Block verification (Algorithm 3)
    4. Difficulty adjustment (DAA)
    5. Chain validation
    """
    print("=" * 70)
    print("POST-QUANTUM PoW CONSENSUS SIMULATION")
    print("Based on Section IV of the paper")
    print("=" * 70)
    
    # Configuration - use small parameters for demo
    # Paper Table II recommends n=12, m=12, q=32 for production
    config = MiningConfig(
        field_size=7,           # F_7 (small for testing)
        n_vars=3,               # 3 variables (n)
        n_equations=3,          # 3 equations (m = n)
        pow_limit=2**254,       # PoWLimit (high for easy demo mining)
        target_block_time=5,    # Target 5s per block
        difficulty_adjustment_window=3  # DAA window
    )
    
    print(f"\nConsensus Parameters:")
    print(f"  Finite Field q: {config.field_size}")
    print(f"  Variables n: {config.n_vars}")
    print(f"  Equations m: {config.n_equations}")
    print(f"  Target Block Time: {config.target_block_time}s")
    print(f"  DAA Window: {config.difficulty_adjustment_window} blocks")
    
    # Create blockchain with genesis
    blockchain = Blockchain(config)
    print(f"\nGenesis block created: {blockchain.get_latest_block().hash().hex()[:32]}...")
    
    # Mine blocks
    print(f"\n{'='*70}")
    print(f"MINING {n_blocks} BLOCKS")
    print("=" * 70)
    
    successful_blocks = 0
    for i in range(n_blocks):
        if verbose:
            print(f"\n--- Block {i+1}/{n_blocks} ---")
        
        block = blockchain.mine_next_block(
            max_nonces=500,
            verbose=verbose
        )
        
        if block:
            success, msg = blockchain.add_block(block)
            if success:
                successful_blocks += 1
                if verbose:
                    print(f"  ✓ {msg}")
            else:
                print(f"  ✗ Failed to add: {msg}")
        else:
            print(f"  ✗ Mining failed for block {i+1}")
    
    # Display results
    blockchain.print_chain()
    
    # Validate entire chain
    print("\n" + "=" * 70)
    print("CHAIN VALIDATION (Algorithm 3)")
    print("=" * 70)
    is_valid, msg = blockchain.validate_chain()
    print(f"Result: {msg}")
    
    # Statistics
    print("\n" + "=" * 70)
    print("CONSENSUS STATISTICS")
    print("=" * 70)
    stats = blockchain.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return blockchain


def demonstrate_verification_speed():
    """
    Demonstrate that verification is much faster than mining.
    
    This is crucial for consensus: miners do expensive work,
    but all nodes can quickly verify the result.
    """
    print("\n" + "=" * 70)
    print("VERIFICATION vs MINING SPEED")
    print("(Key property: verification << mining)")
    print("=" * 70)
    
    config = MiningConfig(
        field_size=7,
        n_vars=3,
        n_equations=3,
        pow_limit=2**250
    )
    
    blockchain = Blockchain(config)
    
    # Mine a block
    print("\nMining a block...")
    start_mine = time.time()
    block = blockchain.mine_next_block(max_nonces=200, verbose=False)
    mine_time = time.time() - start_mine
    
    if block:
        # Verify it many times
        validator = BlockValidator(config)
        
        n_verifications = 100
        start_verify = time.time()
        for _ in range(n_verifications):
            validator.verify_solution(block.header)
        verify_time = time.time() - start_verify
        
        print(f"\nResults:")
        print(f"  Mining time:           {mine_time*1000:.2f} ms")
        print(f"  Verification time (1x): {verify_time/n_verifications*1000:.4f} ms")
        print(f"  Ratio: Mining is {mine_time/(verify_time/n_verifications):.0f}x slower")
        print(f"\n  This asymmetry is essential for PoW consensus!")
    else:
        print("Mining failed")


def demonstrate_tamper_detection():
    """
    Demonstrate that tampering is detected by verification.
    
    If anyone modifies a block's solution, the equations
    won't be satisfied and verification fails.
    """
    print("\n" + "=" * 70)
    print("TAMPER DETECTION DEMONSTRATION")
    print("=" * 70)
    
    config = MiningConfig(
        field_size=7,
        n_vars=3,
        n_equations=3,
        pow_limit=2**252
    )
    
    blockchain = Blockchain(config)
    
    # Mine a few blocks
    print("\nBuilding blockchain...")
    for _ in range(3):
        block = blockchain.mine_next_block(max_nonces=200, verbose=False)
        if block:
            blockchain.add_block(block)
    
    print(f"Chain length: {len(blockchain.chain)} blocks")
    is_valid, msg = blockchain.validate_chain()
    print(f"Initial validation: {msg}")
    
    # Attempt tampering
    print("\n[ATTACK] Modifying block 1's solution...")
    original_solution = blockchain.chain[1].header.solution.copy()
    blockchain.chain[1].header.solution = [1, 2, 3]  # Fake solution
    
    is_valid, msg = blockchain.validate_chain()
    print(f"After tampering: {msg}")
    
    # Restore
    blockchain.chain[1].header.solution = original_solution
    is_valid, msg = blockchain.validate_chain()
    print(f"After restoration: {msg}")
    
    print("\n→ Tampering is detected because fake solutions don't satisfy")
    print("  the deterministically-generated equation system!")


if __name__ == "__main__":
    # Run main consensus simulation
    blockchain = run_consensus_simulation(n_blocks=5, verbose=True)
    
    # Demonstrate key properties
    demonstrate_verification_speed()
    demonstrate_tamper_detection()
    
    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
    print("""
Key Consensus Properties Demonstrated:

1. POST-QUANTUM SECURITY
   - Uses NP-hard multivariate quadratic equations
   - No known quantum speedup (unlike hash-based PoW)

2. MEMORY-HARD MINING  
   - Gröbner basis requires memory proportional to system size
   - ASIC resistance through memory requirements

3. FAST VERIFICATION
   - Mining: Solve equation system (expensive)
   - Verify: Substitute and check (cheap)

4. DIFFICULTY ADJUSTMENT
   - DAA maintains stable block times
   - Target = PoWLimit / Difficulty

5. TAMPER DETECTION
   - Modified solutions fail equation verification
   - Chain integrity preserved
""")
