# Convex Optimization Homework #5-1

1700012751

麦景

----

> 1. Solve (1.1) using CVX by calling different solvers `mosek` and `gurobi`.

为方便书写, 此次作业本人使用了python代码, 托管在[Github平台](https://github.com/magic3007/convex-optimization)上. 其中生成随机数的方式与给出的matlab代码相同, 均使用`mt19937`生成; 程序输出的最终优化值, 稀疏程度, 迭代次数, 运行时间等也相同.具体见[main.py](https://github.com/magic3007/convex-optimization/blob/main/main.py).

我们在python中使用了[`cvxpy`](https://www.cvxpy.org/)来调用CVX, 并分别设置了使用`mosek`和`gurobi`来求解题目的优化问题, 与CVX相关的代码在[cvx_solver_collection.py](https://github.com/magic3007/convex-optimization/blob/main/cvx_solver_collection.py)内. 其中设置使用`mosek`求解器的代码片段如下:

```python
import cvxpy as cp
import numpy as np


# min 0.5 * ||A * x - b||_2^2 + mu * ||x||_{1,2}

def gl_cvx_mosek():
    def foo(x0: np.ndarray, A: np.ndarray, b: np.ndarray, mu, opts):
        # Obtain dimension
        _, n = A.shape
        _, l = b.shape
        # Set up Variable
        x = cp.Variable(shape=(n, l), name='x')
        # Set up Objective
        cost = 0.5 * cp.sum_squares(A @ x - b) + mu * cp.sum(cp.norm(x=x, p=2, axis=1))
        obj = cp.Minimize(expr=cost)
        prob = cp.Problem(objective=obj)
        assert x.shape == x0.shape
        # Set up initial value
        x.value = x0
        
        prob.solve(solver=cp.MOSEK, warm_start=True, verbose=False)
        
        solve_time = prob.solver_stats.solve_time
        num_iters = prob.solver_stats.num_iters
        out = prob.value
        return np.array(x.value), num_iters, out, solve_time

    return foo
```

> First write down an equivalent model of (1.1) which can be solved by calling mosek and gurobi directly, then implement the codes.  

与`mosek`相关的代码在[`mosek_solver_collection.py`](https://github.com/magic3007/convex-optimization/blob/main/mosek_solver_collection.py)中.

对于$||A * x - b||_2^2$, 我们可以把其写成rotated quadratic cone的形式. 不妨设$A * x - b=z$和辅助变量$t^{(1)} \in R$, 则$(\frac{1}{2}, t^{(1)}, z)$构成一个rotated quadratic cone.
$$
2 \times \frac{1}{2} \times t^{(1)} \geq \sum_{i,j}z_{i,j}^2
$$
其在`mosek`中可以写成如下形式:

```python
z = Expr.sub(Expr.mul(A, x), b)
flatten_z = Expr.flatten(z)
 M.constraint(Expr.vstack(0.5, t1, flatten_z), Domain.inRotatedQCone( ))
```

对于$||x||_{1,2}$, 我们可以写成若干个quadratic cone的形式. 设辅助变量$t^{(2)} \in R^n$, 则有
$$
t^{(2)}_i \geq \sqrt{\sum_j x_{i,j}^2}
$$
其在`mosek`中可以写成如下形式:

```python
h = Expr.hstack(t2, x)
# If d=2, it means that each row of a matrix must belong to a cone.
M.constraint(h, Domain.inQCone( ))
```

与`gurobi`相关的代码在[`gurobi_solver_collection.py`](https://github.com/magic3007/convex-optimization/blob/main/gurobi_solver_collection.py)中, 与`mosek`类似, 可写成:

```python
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
```

目前以上四种方法的输出结果如下, 我们可以看到, 总体上看, Guribo在运行时间, 稀疏程度和准确性上均占有一定的优势.

```
2020-11-17 14:34:02,794: INFO  [CVX-Mosek ]: cpu:  0.37, iter:  None, optval: 6.10377E-01, sparisity: 0.120, err-to-exact: 4.02E-05, err-to-cvx-mosek: 0.00E+00, err-to-cvx-gurobi: 3.33E-07
2020-11-17 14:34:04,273: INFO  [CVX-Gurobi]: cpu:  0.73, iter:  None, optval: 6.10377E-01, sparisity: 0.121, err-to-exact: 4.03E-05, err-to-cvx-mosek: 3.33E-07, err-to-cvx-gurobi: 0.00E+00
2020-11-17 14:34:04,700: INFO  [Mosek     ]: cpu:  0.32, iter:    11, optval: 6.10377E-01, sparisity: 0.120, err-to-exact: 4.03E-05, err-to-cvx-mosek: 9.49E-08, err-to-cvx-gurobi: 2.79E-07
2020-11-17 14:34:07,948: INFO  [Gurobi    ]: cpu:  0.23, iter:    12, optval: 6.10378E-01, sparisity: 0.118, err-to-exact: 4.01E-05, err-to-cvx-mosek: 9.17E-07, err-to-cvx-gurobi: 9.85E-07
```



