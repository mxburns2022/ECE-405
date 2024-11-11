import torch
import sympy as sp
from itertools import permutations


class BRIMconfig:
    def __init__(
        self,
        R=310e3,
        C=49e-15,
        scale_start=2.8,
        scale_end=0.4,
        t_step=2.2e-11,
        t_stop=2.2e-6,
    ) -> None:
        self.R: float = R
        self.C: float = C
        self.scale_start: float = scale_start
        self.scale_end: float = scale_end
        self.t_step: float = t_step
        self.t_stop: float = t_stop


def quantize(spin):
    # quantize the spin to -1 and 1
    return torch.sign(spin)


def _calc_energy(spin, J, config):
    # calculate the energy
    _spin = quantize(spin)
    return -0.5 * torch.matmul(_spin.T, torch.matmul(J, _spin)) * config.R * config.maxW

def sign(x):
    return 1 if x > 0 else -1

def unsat_count(spin, clauses):
    unsat_count = len(clauses)
    # <---- YOUR CODE HERE v----->
    for c in clauses:
        for lit in c:
            if sign(lit) == spin[abs(lit)-1].sign():
                unsat_count -= 1
                break
            # unsat_count += 1
    # <---- YOUR CODE HERE ^----->
    return unsat_count


def parse_dimacs_cnf(filename: str) -> tuple[int, list[list[int]]]:
    with open(filename, "r") as file:
        clauses, num_vars = [], 0
        for line in file:
            line = line.strip()
            if not line or line.startswith("c") or "%" in line:
                continue  # skip comments
            if line.startswith("p"):
                parts = line.split()
                num_vars, num_clauses = int(parts[2]), int(parts[3])
                continue
            literals = [int(x) for x in line.split() if int(x) != 0]
            if literals:
                if len(literals) != 3:
                    raise ValueError("Give me a 3-SAT problem.")
                clauses.append(literals)
    return num_vars, clauses


def convert_clauses_to_ising(num_vars, clauses, dtype=torch.float32):
    var_symbols = sp.symbols(f"x1:{num_vars+1}")
    h_coeff = torch.zeros(num_vars, dtype=dtype)
    J_coeff = torch.zeros((num_vars, num_vars), dtype=dtype)
    P_coeff = torch.zeros((num_vars, num_vars, num_vars), dtype=dtype)
    scalar = 0
    for clause in clauses:
        term = 1
        for literal in clause:
            var = var_symbols[abs(literal) - 1]
            term *= (1 - var) if literal > 0 else (1 + var)
        expanded =  1/8 *sp.expand(term) #in HOIM paper, they added 1/8 to make the energy look great
        coeffs = expanded.as_coefficients_dict()
        for monomial, coeff in coeffs.items():
            if monomial == 1:
                scalar += float(coeff)
                continue
            vars_in_monomial = []
            if isinstance(monomial, sp.Symbol):
                vars_in_monomial = [str(monomial)]
            elif isinstance(monomial, sp.Mul):
                vars_in_monomial = sorted([str(arg) for arg in monomial.args])
            else:
                vars_in_monomial = [str(monomial)]
            indices = sorted([int(var[1:]) - 1 for var in vars_in_monomial])
            if len(indices) == 1:
                h_coeff[indices[0]].add_(float(coeff))
            elif len(indices) == 2:
                J_coeff[indices[0], indices[1]].add_(float(coeff))
                J_coeff[indices[1], indices[0]].add_(float(coeff))
            elif len(indices) == 3:
                key = tuple(indices)
                for subkey in permutations(key):
                    P_coeff[subkey].add_(float(coeff))
    return h_coeff, J_coeff, P_coeff, scalar