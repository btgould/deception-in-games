import numpy as np
import scipy.optimize as opt

def random_simplex(n: int) -> np.ndarray:
    """
    Generate a random point in the n-dimensional simplex.
    Args:
        n: Dimension of the simplex.
    Returns:
        A random point in the n-dimensional simplex.
    """

    x = np.random.exponential(scale=1.0, size=n)
    return x / np.sum(x)

num_rows = 3
num_cols = 3
x = random_simplex(num_rows)
y = random_simplex(num_cols)
deception_budget = 2

# Objective function: x^T D y
def objective(D_flat):
    D = D_flat.reshape((num_rows, num_cols))
    return x @ D @ y

# Constraints: For each column j, sum_i |D_ij| <= deception_budget
constraints = []
for j in range(num_cols):
    def col_constraint(D_flat, j=j):
        D = D_flat.reshape((num_rows, num_cols))
        return deception_budget - np.sum(np.abs(D[:, j]))
    constraints.append({'type': 'ineq', 'fun': col_constraint})

# Optionally, you can bound D if you want (e.g., no bounds: (-inf, inf))
bounds = [(-np.inf, np.inf)] * (num_rows * num_cols)

# Initial guess: zero matrix
D0 = np.zeros((num_rows, num_cols)).flatten()

result = opt.minimize(
    objective,
    D0,
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

if result.success:
    D_opt = result.x.reshape((num_rows, num_cols))
    print("Optimal D:\n", D_opt)
    print("Minimum value:", result.fun)
else:
    print("Optimization failed:", result.message)

# Explicitly constructed

outer = np.outer(x, y)
min_val = deception_budget * np.linalg.norm(outer, ord=np.inf)
worst_row = np.argmax(np.sum(np.abs(outer), axis=1))
worst_D = np.zeros_like(outer)
worst_D[worst_row, :] = -deception_budget * np.sign(outer[worst_row, :])

print("Worst-case D:\n", worst_D)
print("Worst-case value:", x @ worst_D @ y)
print(-deception_budget * np.max(x))
