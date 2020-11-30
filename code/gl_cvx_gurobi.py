import cvxpy as cp
import numpy as np


# min 0.5 * ||A * x - b||_2^2 + mu * ||x||_{1,2}

def gl_cvx_gurobi(x0: np.ndarray, A: np.ndarray, b: np.ndarray, mu, opts):
    _, n = A.shape
    _, l = b.shape
    x = cp.Variable(shape=(n, l), name='x')
    cost = 0.5 * cp.sum_squares(A @ x - b) + mu * cp.sum(cp.norm(x=x, p=2, axis=1))
    obj = cp.Minimize(expr=cost)
    prob = cp.Problem(objective=obj)
    assert x.shape == x0.shape
    x.value = x0
    prob.solve(solver=cp.GUROBI, warm_start=True, verbose=False)
    solve_time = prob.solver_stats.solve_time
    num_iters = prob.solver_stats.num_iters
    out = {
        "solve_time": solve_time,
        "fval": prob.value
    }
    return np.array(x.value), num_iters, out
