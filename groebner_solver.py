"""
Gröbner Basis Solver for Multivariate Quadratic Equations

This module implements a Gröbner basis algorithm to solve systems of
multivariate polynomial equations over finite fields.

BACKGROUND ON GRÖBNER BASES:
============================
A Gröbner basis is a particular kind of generating set of a polynomial ideal
that has useful algorithmic properties. The key insight is:

1. Any system of polynomial equations can be transformed into an equivalent
   system (same solutions) that's in "triangular" form
2. This triangular form allows back-substitution to find solutions
3. The transformation is done via the Gröbner basis algorithm

For example, a system like:
    x^2 + y = 0
    x + y^2 = 0

Might be transformed to:
    y^4 - y = 0      (univariate in y)
    x + y^2 = 0      (gives x once we know y)

The F4 algorithm (Faugère, 1999) is an efficient variant that uses linear
algebra (matrix reduction) instead of repeated polynomial division.

WHY THIS IS HARD (AND POST-QUANTUM):
===================================
- Solving multivariate quadratic equations is NP-hard in general
- Gröbner basis computation can be doubly exponential in worst case
- Unlike discrete log or factoring, no known quantum speedup exists
- Memory requirements grow significantly with problem size

This makes it ideal for proof-of-work that resists both classical attacks
(ASICs can't easily parallelize) and quantum attacks.
"""

import hashlib
from typing import List, Tuple, Dict, Set, Optional
from finite_field import FiniteField, FiniteFieldElement
from equation_system import MultivariatePolynomial, EquationSystemGenerator


class Monomial:
    """
    Represents a monomial (product of variables with exponents).
    
    For quadratic systems, monomials are like: 1, x_0, x_1, x_0*x_1, x_0^2, etc.
    
    We represent a monomial as a tuple of exponents: (e_0, e_1, ..., e_{n-1})
    where the monomial is x_0^{e_0} * x_1^{e_1} * ... * x_{n-1}^{e_{n-1}}
    """
    
    def __init__(self, exponents: Tuple[int, ...]):
        self.exponents = tuple(exponents)
        self.n_vars = len(exponents)
        self._degree = sum(exponents)
    
    @classmethod
    def one(cls, n_vars: int) -> 'Monomial':
        """Create the monomial 1 (all exponents zero)"""
        return cls(tuple([0] * n_vars))
    
    @classmethod
    def variable(cls, var_idx: int, n_vars: int) -> 'Monomial':
        """Create a single variable monomial x_i"""
        exp = [0] * n_vars
        exp[var_idx] = 1
        return cls(tuple(exp))
    
    def degree(self) -> int:
        """Total degree of the monomial"""
        return self._degree
    
    def __mul__(self, other: 'Monomial') -> 'Monomial':
        """Multiply two monomials by adding exponents"""
        assert self.n_vars == other.n_vars
        return Monomial(tuple(a + b for a, b in zip(self.exponents, other.exponents)))
    
    def __eq__(self, other):
        return self.exponents == other.exponents
    
    def __hash__(self):
        return hash(self.exponents)
    
    def __lt__(self, other):
        """Graded reverse lexicographic ordering (grevlex)"""
        # First compare by total degree
        if self._degree != other._degree:
            return self._degree < other._degree
        # Then reverse lexicographic (compare from last variable)
        for i in range(self.n_vars - 1, -1, -1):
            if self.exponents[i] != other.exponents[i]:
                return self.exponents[i] > other.exponents[i]  # Note: reversed!
        return False
    
    def __le__(self, other):
        return self == other or self < other
    
    def __repr__(self):
        if self._degree == 0:
            return "1"
        terms = []
        for i, e in enumerate(self.exponents):
            if e == 1:
                terms.append(f"x{i}")
            elif e > 1:
                terms.append(f"x{i}^{e}")
        return "*".join(terms)
    
    def evaluate(self, values: List[FiniteFieldElement]) -> FiniteFieldElement:
        """Evaluate the monomial at given values"""
        result = values[0].field.element(1) if values else None
        field = values[0] if not isinstance(values[0], FiniteFieldElement) else values[0]
        result = FiniteFieldElement(1, field.q if isinstance(field, FiniteFieldElement) else field)
        
        for i, exp in enumerate(self.exponents):
            for _ in range(exp):
                result = result * values[i]
        return result
    
    def divides(self, other: 'Monomial') -> bool:
        """Check if self divides other"""
        return all(a <= b for a, b in zip(self.exponents, other.exponents))
    
    def quotient(self, other: 'Monomial') -> Optional['Monomial']:
        """Return self / other if divisible, None otherwise"""
        if not other.divides(self):
            return None
        return Monomial(tuple(a - b for a, b in zip(self.exponents, other.exponents)))


class SparsePolynomial:
    """
    Sparse representation of a polynomial over a finite field.
    
    Stored as a dictionary: monomial -> coefficient
    This is more efficient for Gröbner basis computation than dense representation.
    """
    
    def __init__(self, field: FiniteField, n_vars: int):
        self.field = field
        self.n_vars = n_vars
        self.terms: Dict[Monomial, FiniteFieldElement] = {}
    
    def set_term(self, mono: Monomial, coeff: FiniteFieldElement):
        """Set coefficient for a monomial"""
        if coeff.is_zero():
            if mono in self.terms:
                del self.terms[mono]
        else:
            self.terms[mono] = coeff
    
    def get_term(self, mono: Monomial) -> FiniteFieldElement:
        """Get coefficient for a monomial"""
        return self.terms.get(mono, self.field.zero())
    
    def add_term(self, mono: Monomial, coeff: FiniteFieldElement):
        """Add to the coefficient of a monomial"""
        current = self.get_term(mono)
        self.set_term(mono, current + coeff)
    
    @classmethod
    def from_multivariate(cls, poly: MultivariatePolynomial) -> 'SparsePolynomial':
        """Convert from MultivariatePolynomial representation"""
        sp = cls(poly.field, poly.n_vars)
        
        # Constant term
        if not poly.constant.is_zero():
            sp.set_term(Monomial.one(poly.n_vars), poly.constant)
        
        # Linear terms
        for i, coeff in poly.linear_terms.items():
            mono = Monomial.variable(i, poly.n_vars)
            sp.set_term(mono, coeff)
        
        # Quadratic terms
        for (i, j), coeff in poly.quadratic_terms.items():
            exp = [0] * poly.n_vars
            exp[i] += 1
            exp[j] += 1
            mono = Monomial(tuple(exp))
            sp.set_term(mono, coeff)
        
        return sp
    
    def is_zero(self) -> bool:
        return len(self.terms) == 0
    
    def leading_monomial(self) -> Optional[Monomial]:
        """Return the leading monomial (largest in term order)"""
        if not self.terms:
            return None
        return max(self.terms.keys())
    
    def leading_coefficient(self) -> Optional[FiniteFieldElement]:
        """Return the coefficient of the leading monomial"""
        lm = self.leading_monomial()
        return self.terms.get(lm) if lm else None
    
    def leading_term(self) -> Tuple[Optional[Monomial], Optional[FiniteFieldElement]]:
        """Return (leading monomial, leading coefficient)"""
        lm = self.leading_monomial()
        lc = self.terms.get(lm) if lm else None
        return lm, lc
    
    def monic(self) -> 'SparsePolynomial':
        """Return a monic version (leading coefficient = 1)"""
        lc = self.leading_coefficient()
        if lc is None or lc.is_zero():
            return self.copy()
        
        inv_lc = lc.inverse()
        result = SparsePolynomial(self.field, self.n_vars)
        for mono, coeff in self.terms.items():
            result.set_term(mono, coeff * inv_lc)
        return result
    
    def copy(self) -> 'SparsePolynomial':
        result = SparsePolynomial(self.field, self.n_vars)
        result.terms = dict(self.terms)
        return result
    
    def __add__(self, other: 'SparsePolynomial') -> 'SparsePolynomial':
        result = self.copy()
        for mono, coeff in other.terms.items():
            result.add_term(mono, coeff)
        return result
    
    def __sub__(self, other: 'SparsePolynomial') -> 'SparsePolynomial':
        result = self.copy()
        for mono, coeff in other.terms.items():
            result.add_term(mono, -coeff)
        return result
    
    def __mul__(self, other) -> 'SparsePolynomial':
        if isinstance(other, FiniteFieldElement):
            result = SparsePolynomial(self.field, self.n_vars)
            for mono, coeff in self.terms.items():
                result.set_term(mono, coeff * other)
            return result
        elif isinstance(other, Monomial):
            result = SparsePolynomial(self.field, self.n_vars)
            for mono, coeff in self.terms.items():
                result.set_term(mono * other, coeff)
            return result
        elif isinstance(other, SparsePolynomial):
            result = SparsePolynomial(self.field, self.n_vars)
            for m1, c1 in self.terms.items():
                for m2, c2 in other.terms.items():
                    result.add_term(m1 * m2, c1 * c2)
            return result
        raise TypeError(f"Cannot multiply SparsePolynomial by {type(other)}")
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __repr__(self):
        if not self.terms:
            return "0"
        sorted_terms = sorted(self.terms.items(), key=lambda x: x[0], reverse=True)
        parts = []
        for mono, coeff in sorted_terms:
            if mono.degree() == 0:
                parts.append(str(coeff.value))
            else:
                parts.append(f"{coeff.value}*{mono}")
        return " + ".join(parts)
    
    def evaluate(self, values: List[FiniteFieldElement]) -> FiniteFieldElement:
        """Evaluate polynomial at given values"""
        result = self.field.zero()
        for mono, coeff in self.terms.items():
            mono_val = self.field.one()
            for i, exp in enumerate(mono.exponents):
                for _ in range(exp):
                    mono_val = mono_val * values[i]
            result = result + coeff * mono_val
        return result


class GrobnerBasisSolver:
    """
    Simplified Gröbner basis solver for small systems.
    
    This implements a basic Buchberger-style algorithm rather than full F4,
    which is sufficient for the small parameter sizes in the paper (n=6 to n=12).
    
    The algorithm:
    1. Start with input polynomials as initial basis
    2. For each pair (f, g), compute the S-polynomial
    3. Reduce S-polynomial against current basis
    4. If remainder is non-zero, add it to basis
    5. Repeat until no new polynomials are added
    6. Extract solutions from the triangular basis
    """
    
    def __init__(self, field: FiniteField, n_vars: int, max_iterations: int = 1000):
        self.field = field
        self.n_vars = n_vars
        self.max_iterations = max_iterations
    
    def s_polynomial(self, f: SparsePolynomial, g: SparsePolynomial) -> SparsePolynomial:
        """
        Compute the S-polynomial of f and g.
        
        S(f,g) = (lcm(LM(f), LM(g)) / LT(f)) * f - (lcm(LM(f), LM(g)) / LT(g)) * g
        
        This cancels the leading terms and often reveals hidden dependencies.
        """
        lm_f, lc_f = f.leading_term()
        lm_g, lc_g = g.leading_term()
        
        if lm_f is None or lm_g is None:
            return SparsePolynomial(self.field, self.n_vars)
        
        # Compute LCM of leading monomials
        lcm_exp = tuple(max(a, b) for a, b in zip(lm_f.exponents, lm_g.exponents))
        lcm_mono = Monomial(lcm_exp)
        
        # Quotients
        quot_f = lcm_mono.quotient(lm_f)
        quot_g = lcm_mono.quotient(lm_g)
        
        # S-polynomial: (1/lc_f) * quot_f * f - (1/lc_g) * quot_g * g
        term1 = (f * quot_f) * lc_f.inverse()
        term2 = (g * quot_g) * lc_g.inverse()
        
        return term1 - term2
    
    def reduce_polynomial(self, f: SparsePolynomial, 
                          basis: List[SparsePolynomial]) -> SparsePolynomial:
        """
        Reduce polynomial f with respect to the basis.
        
        This is polynomial division - repeatedly subtract multiples of basis
        elements to eliminate leading terms when possible.
        """
        result = f.copy()
        changed = True
        max_steps = 1000
        steps = 0
        
        while changed and steps < max_steps:
            changed = False
            steps += 1
            
            lm = result.leading_monomial()
            if lm is None:
                break
            
            for g in basis:
                lm_g = g.leading_monomial()
                if lm_g is None:
                    continue
                
                # Check if LM(g) divides LM(result)
                if lm_g.divides(lm):
                    quot = lm.quotient(lm_g)
                    lc_r = result.leading_coefficient()
                    lc_g = g.leading_coefficient()
                    
                    # result = result - (lc_r / lc_g) * quot * g
                    scalar = lc_r * lc_g.inverse()
                    subtraction = (g * quot) * scalar
                    result = result - subtraction
                    changed = True
                    break
        
        return result
    
    def compute_basis(self, polynomials: List[SparsePolynomial], 
                      verbose: bool = False) -> List[SparsePolynomial]:
        """
        Compute a Gröbner basis for the given polynomials.
        
        Returns:
            List of polynomials forming a Gröbner basis
        """
        # Initialize basis with monic versions of input polynomials
        basis = [p.monic() for p in polynomials if not p.is_zero()]
        
        if verbose:
            print(f"Starting Gröbner basis computation with {len(basis)} polynomials")
        
        # Pairs to process
        pairs = [(i, j) for i in range(len(basis)) for j in range(i+1, len(basis))]
        
        iteration = 0
        while pairs and iteration < self.max_iterations:
            iteration += 1
            i, j = pairs.pop(0)
            
            if i >= len(basis) or j >= len(basis):
                continue
            
            # Compute S-polynomial
            s = self.s_polynomial(basis[i], basis[j])
            
            # Reduce against current basis
            s_reduced = self.reduce_polynomial(s, basis)
            
            # If non-zero remainder, add to basis
            if not s_reduced.is_zero():
                s_reduced = s_reduced.monic()
                new_idx = len(basis)
                basis.append(s_reduced)
                
                # Add new pairs
                for k in range(new_idx):
                    pairs.append((k, new_idx))
                
                if verbose and iteration % 10 == 0:
                    print(f"  Iteration {iteration}: basis size = {len(basis)}")
        
        if verbose:
            print(f"Finished after {iteration} iterations, basis size = {len(basis)}")
        
        return basis
    
    def solve_system(self, polynomials: List[SparsePolynomial],
                     verbose: bool = False) -> List[List[FiniteFieldElement]]:
        """
        Solve the polynomial system by computing Gröbner basis and back-substitution.
        
        Returns:
            List of solutions, where each solution is [x_0, x_1, ..., x_{n-1}]
        """
        if verbose:
            print("Computing Gröbner basis...")
        
        basis = self.compute_basis(polynomials, verbose)
        
        if verbose:
            print("Extracting solutions via enumeration...")
        
        # For small fields, we can enumerate all possible solutions
        # and check which ones satisfy all equations
        solutions = []
        
        # Enumerate all points in F_q^n
        def enumerate_points(depth: int, current: List[FiniteFieldElement]):
            if depth == self.n_vars:
                # Check if this point is a solution
                is_solution = True
                for p in basis:
                    if not p.evaluate(current).is_zero():
                        is_solution = False
                        break
                if is_solution:
                    solutions.append(list(current))
                return
            
            for val in range(self.field.q):
                current.append(self.field.element(val))
                enumerate_points(depth + 1, current)
                current.pop()
        
        enumerate_points(0, [])
        
        if verbose:
            print(f"Found {len(solutions)} solution(s)")
        
        return solutions


def solve_equations(equations: List[MultivariatePolynomial],
                    verbose: bool = False) -> List[List[FiniteFieldElement]]:
    """
    High-level interface to solve a system of multivariate equations.
    
    Args:
        equations: List of MultivariatePolynomial objects
        verbose: Whether to print progress
    
    Returns:
        List of solutions
    """
    if not equations:
        return []
    
    field = equations[0].field
    n_vars = equations[0].n_vars
    
    # Convert to sparse representation
    sparse_polys = [SparsePolynomial.from_multivariate(eq) for eq in equations]
    
    # Solve
    solver = GrobnerBasisSolver(field, n_vars)
    return solver.solve_system(sparse_polys, verbose)


# Test the solver
if __name__ == "__main__":
    print("Testing Gröbner Basis Solver")
    print("=" * 60)
    
    # Create a simple system with known solution
    # x0 + x1 = 3
    # x0 * x1 = 2
    # In F_31: solutions should be roots of t^2 - 3t + 2 = (t-1)(t-2) = 0
    # So (x0, x1) = (1, 2) or (2, 1)
    
    field = FiniteField(31)
    n_vars = 2
    
    # Create polynomials manually
    p1 = SparsePolynomial(field, n_vars)  # x0 + x1 - 3
    p1.set_term(Monomial.variable(0, n_vars), field.element(1))
    p1.set_term(Monomial.variable(1, n_vars), field.element(1))
    p1.set_term(Monomial.one(n_vars), field.element(-3))
    
    p2 = SparsePolynomial(field, n_vars)  # x0 * x1 - 2
    exp_prod = (1, 1)
    p2.set_term(Monomial(exp_prod), field.element(1))
    p2.set_term(Monomial.one(n_vars), field.element(-2))
    
    print(f"Equation 1: {p1}")
    print(f"Equation 2: {p2}")
    
    solver = GrobnerBasisSolver(field, n_vars)
    solutions = solver.solve_system([p1, p2], verbose=True)
    
    print(f"\nSolutions found:")
    for sol in solutions:
        print(f"  ({sol[0].value}, {sol[1].value})")
        # Verify
        v1 = p1.evaluate(sol)
        v2 = p2.evaluate(sol)
        print(f"    Verification: p1={v1.value}, p2={v2.value}")
    
    print("\n" + "=" * 60)
    print("Testing with random quadratic system (small parameters)")
    print("=" * 60)
    
    # Now test with a randomly generated system
    import hashlib
    
    q = 7  # Very small field for quick testing
    n = 3  # 3 variables
    m = 3  # 3 equations
    
    field = FiniteField(q)
    generator = EquationSystemGenerator(field, n, m)
    
    prev_hash = hashlib.sha256(b"test").digest()
    seed = generator.generate_seed(prev_hash, 42)
    
    equations = generator.generate_system(seed)
    print(f"\nGenerated {m} equations in {n} variables over F_{q}:")
    for i, eq in enumerate(equations):
        print(f"  f_{i}: {eq}")
    
    solutions = solve_equations(equations, verbose=True)
    print(f"\nFound {len(solutions)} solution(s)")
    for sol in solutions:
        vals = [s.value for s in sol]
        print(f"  Solution: {vals}")
        # Verify
        for i, eq in enumerate(equations):
            result = eq.evaluate(sol)
            print(f"    f_{i}({vals}) = {result.value}")
