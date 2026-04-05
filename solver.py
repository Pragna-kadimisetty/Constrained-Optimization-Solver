import sympy as sp
import numpy as np
from scipy.optimize import minimize

# NEW IMPORTS (for GNN)
import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# GNN MODEL (simple version)
# -----------------------------
class SimpleGNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleGNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# -----------------------------
# MAIN SOLVER CLASS
# -----------------------------
class ConstrainedOptimizationSolver:

    def __init__(self):
        # Initialize GNN model
        self.model = SimpleGNN(input_size=2, hidden_size=16, output_size=2)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()

    # -----------------------------
    # EXISTING LAGRANGE SOLVER
    # -----------------------------
    def lagrange_solver(self, objective_expr, constraint_expr, variables):

        vars_symbols = sp.symbols(variables)
        lam = sp.symbols('lam')

        f = sp.sympify(objective_expr)
        g = sp.sympify(constraint_expr)

        L = f + lam * g

        equations = []

        for v in vars_symbols:
            equations.append(sp.diff(L, v))

        equations.append(sp.diff(L, lam))

        solution = sp.solve(equations, (*vars_symbols, lam), dict=True)

        return solution

    # -----------------------------
    # EXISTING KKT SOLVER
    # -----------------------------
    def kkt_solver(self, objective_func, initial_guess, constraints):

        result = minimize(
            objective_func,
            initial_guess,
            constraints=constraints
        )

        return result

    # -----------------------------
    # NEW: DATA GENERATION
    # -----------------------------
    def generate_training_data(self, num_samples=50):
        """
        Generate simple optimization problems and solve using Lagrange
        """

        X = []
        Y = []

        for _ in range(num_samples):

            # Random constraint: x + y = c
            c = np.random.uniform(1, 10)

            objective = "x**2 + y**2"
            constraint = f"x + y - {c}"

            sol = self.lagrange_solver(objective, constraint, "x y")

            if sol:
                sol = sol[0]
                x_val = float(sol[sp.Symbol('x')])
                y_val = float(sol[sp.Symbol('y')])

                # Input = constraint value
                X.append([c, c])

                # Output = optimal solution
                Y.append([x_val, y_val])

        return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

    # -----------------------------
    # NEW: TRAIN GNN
    # -----------------------------
    def train_gnn(self, epochs=100):

        X, Y = self.generate_training_data()

        for epoch in range(epochs):
            self.optimizer.zero_grad()

            outputs = self.model(X)
            loss = self.criterion(outputs, Y)

            loss.backward()
            self.optimizer.step()

        print("GNN Training Completed")

    # -----------------------------
    # NEW: AI SOLVER
    # -----------------------------
    def gnn_solver(self, constraint_value):
        """
        Predict solution using trained GNN
        """

        inp = torch.tensor([[constraint_value, constraint_value]], dtype=torch.float32)

        with torch.no_grad():
            prediction = self.model(inp)

        return prediction.numpy()


# -----------------------------
# RUN EVERYTHING
# -----------------------------
solver = ConstrainedOptimizationSolver()

print("----- TRAINING GNN -----")
solver.train_gnn()

print("\n----- LAGRANGE SOLVER -----")
objective = "x**2 + y**2"
constraint = "x + y - 4"

solution = solver.lagrange_solver(objective, constraint, "x y")

print("Exact Solution:", solution)

print("\n----- GNN PREDICTION -----")
pred = solver.gnn_solver(4)

print("Predicted Solution (AI):", pred)