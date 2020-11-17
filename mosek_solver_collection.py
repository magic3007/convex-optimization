from mosek.fusion import *
import numpy as np


# min 0.5 * ||A * x - b||_2^2 + mu * ||x||_{1,2}


def gl_mosek():
    def foo(x0: np.ndarray, A: np.ndarray, b: np.ndarray, mu, opts):
        _, n = A.shape
        _, l = b.shape
        with Model('gl_mosek') as M:
            x = M.variable('x', [ n, l ], Domain.unbounded( ))
            t1 = M.variable('t1')
            t2 = M.variable('t2', n, Domain.unbounded( ))

            z = Expr.sub(Expr.mul(A, x), b)
            flatten_z = Expr.flatten(z)
            M.constraint(Expr.vstack(0.5, t1, flatten_z), Domain.inRotatedQCone( ))

            h = Expr.hstack(t2, x)
            # If d=2, it means that each row of a matrix must belong to a cone.
            M.constraint(h, Domain.inQCone( ))

            cost1 = Expr.mul(0.5, t1)
            cost2 = Expr.mul(mu, Expr.sum(t2))
            cost = Expr.add(cost1, cost2)

            M.objective(ObjectiveSense.Minimize, cost)
            # Set values for an initial solution
            # x.setLevel(x0.flatten( ).tolist( ))
            M.solve( )

            solve_time = M.getSolverDoubleInfo("optimizerTime")
            num_iters = M.getSolverIntInfo("intpntIter")
            out = M.primalObjValue( )
            rv = np.array(x.level( )).reshape(n, l)

            return rv, num_iters, out, solve_time

    return foo
