import numpy as np
import gurobipy as gp


# min 0.5 * ||A * x - b||_2^2 + mu * ||x||_{1,2}

def gl_gurobi(x0: np.ndarray, A: np.ndarray, b: np.ndarray, mu, opts):
    _, n = A.shape
    m, l = b.shape
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start( )
        # Build model m here
        with gp.Model('Gurobi', env=env) as M:
            # The default lower bound is 0.
            x = M.addMVar(shape=(n, l), name='x', lb=-gp.GRB.INFINITY)
            t2 = M.addMVar(shape=(n,), name='t2')
            z = M.addMVar(shape=b.shape, name='z', lb=-gp.GRB.INFINITY)

            cost = mu * t2.sum( )
            for i in range(m):
                cost += z[ i, : ] @ z[ i, : ] * 0.5
            M.setObjective(cost)

            M.addConstrs(z[ :, i ] + b[ :, i ] == A @ x[ :, i ] for i in range(l))
            M.addConstrs(t2[ i ] @ t2[ i ] >= x[ i, : ] @ x[ i, : ] for i in range(n))

            M.optimize( )

            solve_time = M.Runtime
            fval = M.objVal
            num_iters = M.BarIterCount  # Number of barrier iterations performed
            rv = x.X

            out = {
                "fval": fval,
                "solve_time": solve_time
            }
            return rv, num_iters, out
