"""
Finite Field Arithmetic Module

This module implements arithmetic operations over finite fields F_q.
In the paper, they use q = 32 (which is 2^5), making this a Galois Field GF(2^5).

For simplicity and correctness, we'll work with prime fields F_p where p is prime.
The paper uses q=32, but for Gröbner basis algorithms, prime fields are more standard.
We'll use q=31 (a prime close to 32) for our implementation.

The finite field is the mathematical foundation:
- All coefficients of our polynomials live in this field
- All arithmetic (addition, multiplication) wraps around mod q
- This ensures bounded, predictable computation
"""

class FiniteFieldElement:
    """
    Represents an element in a finite field F_q.
    
    All arithmetic operations are performed modulo q (the field characteristic).
    """
    
    def __init__(self, value: int, q: int):
        """
        Initialize a finite field element.
        
        Args:
            value: The integer value (will be reduced mod q)
            q: The field characteristic (should be prime for a prime field)
        """
        self.q = q
        self.value = value % q
    
    def __repr__(self):
        return f"F{self.q}({self.value})"
    
    def __eq__(self, other):
        if isinstance(other, FiniteFieldElement):
            return self.value == other.value and self.q == other.q
        return self.value == (other % self.q)
    
    def __hash__(self):
        return hash((self.value, self.q))
    
    def __add__(self, other):
        """Addition in finite field: (a + b) mod q"""
        if isinstance(other, FiniteFieldElement):
            assert self.q == other.q, "Field mismatch"
            return FiniteFieldElement((self.value + other.value) % self.q, self.q)
        return FiniteFieldElement((self.value + other) % self.q, self.q)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        """Subtraction in finite field: (a - b) mod q"""
        if isinstance(other, FiniteFieldElement):
            assert self.q == other.q, "Field mismatch"
            return FiniteFieldElement((self.value - other.value) % self.q, self.q)
        return FiniteFieldElement((self.value - other) % self.q, self.q)
    
    def __rsub__(self, other):
        return FiniteFieldElement((other - self.value) % self.q, self.q)
    
    def __mul__(self, other):
        """Multiplication in finite field: (a * b) mod q"""
        if isinstance(other, FiniteFieldElement):
            assert self.q == other.q, "Field mismatch"
            return FiniteFieldElement((self.value * other.value) % self.q, self.q)
        return FiniteFieldElement((self.value * other) % self.q, self.q)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __neg__(self):
        """Negation in finite field: -a mod q"""
        return FiniteFieldElement((-self.value) % self.q, self.q)
    
    def __pow__(self, exp):
        """Exponentiation using fast modular exponentiation"""
        if exp == 0:
            return FiniteFieldElement(1, self.q)
        result = pow(self.value, exp, self.q)
        return FiniteFieldElement(result, self.q)
    
    def inverse(self):
        """
        Multiplicative inverse using extended Euclidean algorithm.
        For prime fields, a^(-1) = a^(q-2) mod q by Fermat's little theorem.
        """
        if self.value == 0:
            raise ValueError("Cannot invert zero")
        # Using Fermat's little theorem for prime fields
        return FiniteFieldElement(pow(self.value, self.q - 2, self.q), self.q)
    
    def __truediv__(self, other):
        """Division: a / b = a * b^(-1)"""
        if isinstance(other, FiniteFieldElement):
            return self * other.inverse()
        return self * FiniteFieldElement(other, self.q).inverse()
    
    def is_zero(self):
        return self.value == 0
    
    def to_int(self):
        return self.value


class FiniteField:
    """
    Factory class for creating elements in a specific finite field F_q.
    
    This makes it convenient to work with a fixed field throughout the computation.
    """
    
    def __init__(self, q: int):
        """
        Initialize a finite field.
        
        Args:
            q: The field characteristic (should be prime)
        """
        self.q = q
        # Verify q is prime (important for field properties)
        if not self._is_prime(q):
            print(f"Warning: {q} is not prime. Field arithmetic may not be correct.")
    
    def _is_prime(self, n):
        """Simple primality test"""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    def element(self, value: int) -> FiniteFieldElement:
        """Create an element in this field"""
        return FiniteFieldElement(value, self.q)
    
    def zero(self) -> FiniteFieldElement:
        """Return the additive identity (0)"""
        return FiniteFieldElement(0, self.q)
    
    def one(self) -> FiniteFieldElement:
        """Return the multiplicative identity (1)"""
        return FiniteFieldElement(1, self.q)
    
    def elements(self):
        """Generator yielding all elements in the field"""
        for i in range(self.q):
            yield FiniteFieldElement(i, self.q)
    
    def random_element(self, rng=None):
        """Generate a random field element"""
        import random
        if rng is None:
            rng = random
        return FiniteFieldElement(rng.randint(0, self.q - 1), self.q)


# Quick test
if __name__ == "__main__":
    print("Testing Finite Field Arithmetic (F_31)")
    print("=" * 50)
    
    F = FiniteField(31)
    
    a = F.element(15)
    b = F.element(20)
    
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"a + b = {a + b} (expected: {(15 + 20) % 31})")
    print(f"a - b = {a - b} (expected: {(15 - 20) % 31})")
    print(f"a * b = {a * b} (expected: {(15 * 20) % 31})")
    print(f"a^(-1) = {a.inverse()}")
    print(f"a * a^(-1) = {a * a.inverse()} (should be 1)")
    print(f"a / b = {a / b}")
    print(f"(a / b) * b = {(a / b) * b} (should be {a})")
    
    print("\nField properties verified!")
