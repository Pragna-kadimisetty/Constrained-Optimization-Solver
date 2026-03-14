import sympy as sp
import numpy as np
from scipy.optimize import minimize

class ConstrainedOptimizationSolver:

    def lagrange_solver(self, objective_expr, constraint_expr, variables):
        """
        Solve equality constrained optimization using Lagrange Multipliers
        """

        # Create symbols
        vars_symbols = sp.symbols(variables)
        lam = sp.symbols('lam')

        # Convert expressions
        f = sp.sympify(objective_expr)
        g = sp.sympify(constraint_expr)

        # Build Lagrangian
        L = f + lam * g

        equations = []

        # Partial derivatives
        for v in vars_symbols:
            equations.append(sp.diff(L, v))

        equations.append(sp.diff(L, lam))

        # Solve system
        solution = sp.solve(equations, (*vars_symbols, lam), dict=True)

        return solution


    def kkt_solver(self, objective_func, initial_guess, constraints):
        """
        Solve inequality constrained optimization using KKT (SciPy)
        """

        result = minimize(
            objective_func,
            initial_guess,
            constraints=constraints
        )

        return result


# -----------------------------
# Example Usage
# -----------------------------

solver = ConstrainedOptimizationSolver()

print("----- LAGRANGE MULTIPLIER SOLVER -----")

objective = "x**2 + y**2"
constraint = "x + y - 1"

solution = solver.lagrange_solver(objective, constraint, "x y")

print("Solutions:")
for sol in solution:
    print(sol)


print("\n----- KKT SOLVER (INEQUALITY) -----")

# Objective function
def objective(v):
    x, y = v
    return x**2 + y**2


# inequality constraint: x + y >= 1
constraint = {
    'type': 'ineq',
    'fun': lambda v: v[0] + v[1] - 1
}

result = solver.kkt_solver(objective, [0, 0], [constraint])

print("Optimal point:", result.x)
print("Minimum value:", result.fun)