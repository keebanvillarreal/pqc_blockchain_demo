"""
Multivariate Quadratic Equation System Generator

This module generates systems of multivariate quadratic equations as specified
in the paper. The key insight is that solving such systems is an NP-hard problem,
making it suitable as a proof-of-work puzzle that's resistant to quantum attacks.

The paper specifies:
- n variables: x_1, x_2, ..., x_n
- m equations: f_1, f_2, ..., f_m (typically m = n for a unique solution)
- Each equation is quadratic: contains terms x_i*x_j, x_i, and constants
- All coefficients come from a finite field F_q
- Coefficients are deterministically generated from a seed (using PRNG)

This determinism is crucial: 
- All miners generate the SAME equation system from the same seed
- Anyone can verify a solution by regenerating the equations
"""

import hashlib
import struct
from typing import List, Tuple, Dict, Optional
from finite_field import FiniteField, FiniteFieldElement


class DeterministicPRNG:
    """
    Deterministic Pseudo-Random Number Generator based on SHA-256.
    
    This is critical for consensus: given the same seed, all nodes
    must generate the exact same sequence of random numbers.
    
    The paper uses: Random_i = PRNG(Seed, i) / q
    We implement this by repeatedly hashing and extracting integers.
    """
    
    def __init__(self, seed: bytes):
        """
        Initialize PRNG with a seed (typically derived from block hash + nonce).
        
        Args:
            seed: Initial seed bytes (from SHA256 of block data)
        """
        self.seed = seed
        self.counter = 0
        self.buffer = b''
        self.buffer_pos = 0
    
    def _generate_bytes(self, n: int) -> bytes:
        """Generate n random bytes deterministically"""
        result = b''
        while len(result) < n:
            # Hash: SHA256(seed || counter)
            data = self.seed + struct.pack('>Q', self.counter)
            hash_output = hashlib.sha256(data).digest()
            result += hash_output
            self.counter += 1
        return result[:n]
    
    def randint(self, min_val: int, max_val: int) -> int:
        """
        Generate a random integer in [min_val, max_val].
        
        Uses rejection sampling to ensure uniform distribution.
        """
        range_size = max_val - min_val + 1
        
        # Determine how many bytes we need
        bytes_needed = (range_size.bit_length() + 7) // 8
        
        # Rejection sampling for uniform distribution
        while True:
            rand_bytes = self._generate_bytes(bytes_needed)
            value = int.from_bytes(rand_bytes, 'big')
            
            # Rejection sampling to avoid modulo bias
            max_valid = (256 ** bytes_needed // range_size) * range_size
            if value < max_valid:
                return min_val + (value % range_size)
    
    def random_field_element(self, field: FiniteField) -> FiniteFieldElement:
        """Generate a random element in the given finite field"""
        return field.element(self.randint(0, field.q - 1))


class MultivariatePolynomial:
    """
    Represents a multivariate quadratic polynomial over a finite field.
    
    A polynomial is stored as:
    - quadratic_terms: Dict[(i,j) -> coefficient] for x_i * x_j terms
    - linear_terms: Dict[i -> coefficient] for x_i terms  
    - constant: The constant term
    
    Example: 3*x1*x2 + 2*x1 + 5 would be stored as:
    - quadratic_terms = {(0,1): 3}
    - linear_terms = {0: 2}
    - constant = 5
    """
    
    def __init__(self, field: FiniteField, n_vars: int):
        """
        Initialize an empty polynomial.
        
        Args:
            field: The finite field for coefficients
            n_vars: Number of variables (x_0, x_1, ..., x_{n-1})
        """
        self.field = field
        self.n_vars = n_vars
        self.quadratic_terms: Dict[Tuple[int, int], FiniteFieldElement] = {}
        self.linear_terms: Dict[int, FiniteFieldElement] = {}
        self.constant: FiniteFieldElement = field.zero()
    
    def set_quadratic(self, i: int, j: int, coeff: FiniteFieldElement):
        """Set coefficient for x_i * x_j term (ensures i <= j for canonical form)"""
        if i > j:
            i, j = j, i
        if not coeff.is_zero():
            self.quadratic_terms[(i, j)] = coeff
        elif (i, j) in self.quadratic_terms:
            del self.quadratic_terms[(i, j)]
    
    def set_linear(self, i: int, coeff: FiniteFieldElement):
        """Set coefficient for x_i term"""
        if not coeff.is_zero():
            self.linear_terms[i] = coeff
        elif i in self.linear_terms:
            del self.linear_terms[i]
    
    def set_constant(self, coeff: FiniteFieldElement):
        """Set the constant term"""
        self.constant = coeff
    
    def evaluate(self, values: List[FiniteFieldElement]) -> FiniteFieldElement:
        """
        Evaluate the polynomial at given variable values.
        
        Args:
            values: List of field elements [x_0, x_1, ..., x_{n-1}]
        
        Returns:
            The polynomial evaluated at these values
        """
        assert len(values) == self.n_vars, f"Expected {self.n_vars} values, got {len(values)}"
        
        result = self.constant
        
        # Add linear terms
        for i, coeff in self.linear_terms.items():
            result = result + coeff * values[i]
        
        # Add quadratic terms
        for (i, j), coeff in self.quadratic_terms.items():
            result = result + coeff * values[i] * values[j]
        
        return result
    
    def __repr__(self):
        terms = []
        
        # Quadratic terms
        for (i, j), coeff in sorted(self.quadratic_terms.items()):
            if i == j:
                terms.append(f"{coeff.value}*x{i}^2")
            else:
                terms.append(f"{coeff.value}*x{i}*x{j}")
        
        # Linear terms
        for i, coeff in sorted(self.linear_terms.items()):
            terms.append(f"{coeff.value}*x{i}")
        
        # Constant
        if not self.constant.is_zero() or not terms:
            terms.append(str(self.constant.value))
        
        return " + ".join(terms) if terms else "0"


class EquationSystemGenerator:
    """
    Generates a system of multivariate quadratic equations from a seed.
    
    This is the core of the PoW puzzle generation as described in the paper:
    1. Take Seed = SHA256(SHA256(Block_{i-1}) | Nonce)
    2. Use PRNG(Seed, i) to generate coefficients
    3. Build m equations in n variables
    
    The number of random coefficients needed (from paper):
    - Quadratic: n*(n+1)/2 coefficients per equation (x_i*x_j where i <= j)
    - Linear: n coefficients per equation
    - Constant: 1 per equation
    - Total per equation: n*(n+1)/2 + n + 1 = (n^2 + 3n + 2)/2
    - Total for m equations: m * (n^2 + 3n + 2)/2
    """
    
    def __init__(self, field: FiniteField, n_vars: int, n_equations: int):
        """
        Initialize the equation system generator.
        
        Args:
            field: The finite field F_q
            n_vars: Number of variables (n)
            n_equations: Number of equations (m), typically m = n
        """
        self.field = field
        self.n_vars = n_vars
        self.n_equations = n_equations
    
    def generate_seed(self, prev_block_hash: bytes, nonce: int) -> bytes:
        """
        Generate the seed for equation generation.
        
        From paper: Seed = SHA256(SHA256(Block_{i-1}) | Nonce)
        
        Args:
            prev_block_hash: Hash of the previous block
            nonce: Current nonce being tried
        
        Returns:
            32-byte seed for PRNG
        """
        # First hash of the previous block hash
        inner_hash = hashlib.sha256(prev_block_hash).digest()
        
        # Concatenate with nonce and hash again
        nonce_bytes = struct.pack('>Q', nonce)  # 8-byte big-endian
        combined = inner_hash + nonce_bytes
        
        return hashlib.sha256(combined).digest()
    
    def generate_system(self, seed: bytes) -> List[MultivariatePolynomial]:
        """
        Generate the complete system of equations from a seed.
        
        Args:
            seed: The seed bytes (from generate_seed)
        
        Returns:
            List of m MultivariatePolynomial objects
        """
        prng = DeterministicPRNG(seed)
        equations = []
        
        for eq_idx in range(self.n_equations):
            poly = MultivariatePolynomial(self.field, self.n_vars)
            
            # Generate quadratic coefficients (x_i * x_j where i <= j)
            # This gives n*(n+1)/2 terms
            for i in range(self.n_vars):
                for j in range(i, self.n_vars):
                    coeff = prng.random_field_element(self.field)
                    poly.set_quadratic(i, j, coeff)
            
            # Generate linear coefficients
            for i in range(self.n_vars):
                coeff = prng.random_field_element(self.field)
                poly.set_linear(i, coeff)
            
            # Generate constant term
            const = prng.random_field_element(self.field)
            poly.set_constant(const)
            
            equations.append(poly)
        
        return equations
    
    def verify_solution(self, equations: List[MultivariatePolynomial], 
                        solution: List[FiniteFieldElement]) -> bool:
        """
        Verify that a solution satisfies all equations.
        
        A valid solution means f_k(x_1, ..., x_n) = 0 for all k.
        
        Args:
            equations: The system of equations
            solution: Proposed solution [x_0, x_1, ..., x_{n-1}]
        
        Returns:
            True if all equations evaluate to 0
        """
        for eq in equations:
            result = eq.evaluate(solution)
            if not result.is_zero():
                return False
        return True


def count_coefficients(n: int, m: int) -> int:
    """
    Calculate total number of coefficients needed.
    
    From paper: n^2 * m + n * m + 2m for the general formula,
    but more precisely for our implementation:
    - Quadratic: m * n*(n+1)/2
    - Linear: m * n  
    - Constant: m
    """
    quadratic = m * n * (n + 1) // 2
    linear = m * n
    constant = m
    return quadratic + linear + constant


# Test the equation generation
if __name__ == "__main__":
    print("Testing Equation System Generation")
    print("=" * 60)
    
    # Use parameters similar to paper's test case
    q = 31  # Prime field (paper uses q=32, we use 31 for prime field)
    n = 4   # 4 variables (smaller for testing)
    m = 4   # 4 equations
    
    field = FiniteField(q)
    generator = EquationSystemGenerator(field, n, m)
    
    # Simulate a previous block hash and nonce
    prev_hash = hashlib.sha256(b"genesis_block").digest()
    nonce = 12345
    
    # Generate seed
    seed = generator.generate_seed(prev_hash, nonce)
    print(f"Seed (hex): {seed.hex()[:32]}...")
    
    # Generate equations
    print(f"\nGenerating {m} equations in {n} variables over F_{q}")
    print(f"Total coefficients needed: {count_coefficients(n, m)}")
    
    equations = generator.generate_system(seed)
    
    print("\nGenerated Equations:")
    for i, eq in enumerate(equations):
        print(f"  f_{i}: {eq}")
    
    # Verify determinism: same seed should give same equations
    print("\nVerifying determinism...")
    equations2 = generator.generate_system(seed)
    same = all(str(e1) == str(e2) for e1, e2 in zip(equations, equations2))
    print(f"Same seed produces same equations: {same}")
    
    # Test with a random point (likely not a solution)
    print("\nEvaluating at random point:")
    random_point = [field.element(i) for i in range(n)]
    print(f"  Point: {[p.value for p in random_point]}")
    for i, eq in enumerate(equations):
        result = eq.evaluate(random_point)
        print(f"  f_{i}({[p.value for p in random_point]}) = {result.value}")
