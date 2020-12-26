# Convex Optimization Homework #5

1700012751

麦景

----

[TOC]

## FYI

本次作业使用python完成, 并托管在[Github](https://github.com/magic3007/convex-optimization)上. 关于Python环境配置和所需安装包等, 请见[code/README.md](https://github.com/magic3007/convex-optimization/blob/main/code/README.md).

测试算例的生成代码如下, 其形式与默认随机种子与给定的matlab代码完全一致, 但是由于matlab和python在相同随机数生成器下生成正态分布的方式可能不完全相同, 因此生成的测试算例可能不同. 接下来同样会测试在其他种子下的数值表现情况.

```python
def gen_data(seed=97006855):
    n, m, l = 512, 256, 2
    mu = 1e-2
    generator = random.Generator(random.MT19937(seed=seed))
    A = generator.standard_normal(size=(m, n))
    k = round(n * 0.1)
    p = generator.permutation(n)[ :k ]
    u = np.zeros(shape=(n, l))
    u[ p, : ] = generator.standard_normal(size=(k, l))  # ground truth
    b = np.matmul(A, u)
    x0 = generator.standard_normal(size=(n, l))
    errfun = lambda x1, x2: norm(x1 - x2, 'fro') / (1 + norm(x1, 'fro'))
    errfun_exact = lambda x: norm(x - u, 'fro') / (1 + norm(x, 'fro'))
    sparsity = lambda x: np.sum(np.abs(x) > 1e-6 * np.max(np.abs(x))) / (n * l)
    return n, m, l, mu, A, b, u, x0, errfun, errfun_exact, sparsity
```

## Problem #1

> Solve (1.1) using CVX by calling different solvers `mosek` and `gurobi`.

我们在python中使用了[`cvxpy`](https://www.cvxpy.org/)来调用CVX, 并分别设置了使用`mosek`和`gurobi`来求解题目的优化问题, 相关代码分别为[gl_cvx_mosek.py](https://github.com/magic3007/convex-optimization/blob/main/code/gl_cvx_mosek.py)和[gl_cvx_gurobi.py](https://github.com/magic3007/convex-optimization/blob/main/code/gl_cvx_gurobi.py).

## Problem #2

> First write down an equivalent model of (1.1) which can be solved by calling mosek and gurobi directly, then implement the codes.  

使用`mosek` solver的代码在[gl_mosek.py](https://github.com/magic3007/convex-optimization/blob/main/code/gl_mosek.py)中. 对于$||A * x - b||_2^2$, 我们可以把其写成rotated quadratic cone的形式. 不妨设$A * x - b=z$和辅助变量$t^{(1)} \in R$, 则$(\frac{1}{2}, t^{(1)}, z)$构成一个rotated quadratic cone.

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
其核心代码为:

```python
h = Expr.hstack(t2, x)
# If d=2, it means that each row of a matrix must belong to a cone.
M.constraint(h, Domain.inQCone( ))
```

使用`gurobi` solver的代码在[`gl_gurobi.py`](https://github.com/magic3007/convex-optimization/blob/main/code/gl_gurobi.py)中, 与`mosek`类似, 可写成:

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

目前以上四种方法在默认随机种子下的输出结果如下:

|  solver  | cpu |iter |  optval   |sparsity|err-to-exact|err-to-cvx-mosek|err-to-cvx-gurobi|
|----------|-----|-----|-----------|--------|------------|----------------|-----------------|
|CVX-Mosek | 0.31|   -1|6.10377E-01|0.1201  |4.02E-05    |0.00E+00        |3.33E-07         |
|CVX-Gurobi| 0.69|   -1|6.10377E-01|0.1211  |4.03E-05    |3.33E-07        |0.00E+00         |
|Mosek     | 0.29|   11|6.10377E-01|0.1201  |4.03E-05    |9.49E-08        |2.79E-07         |
|Gurobi    | 0.23|   12|6.10378E-01|0.1182  |4.01E-05    |9.17E-07        |9.85E-07         |

我们可以看到, 总体上看, Guribo在运行时间, 稀疏程度和恢复效果上均占有一定的优势.

## Problem #3 (a) & (b)

> (a) Subgradient method for the primal problem.
>
> (b) Gradient method for the smoothed primal problem.

 首先给出这两个问题的数学形式. 对于问题(a), 设目标函数为$f(x)=\frac{1}{2}\left \| Ax-b \right \|_F^2+\mu \left \| x \right \|_{1,2}$. 其中$\left \| Ax-b \right \|_F^2$部分可导, 其次梯度为$\partial \left \| Ax-b \right \|_F^2 = \{A^T(Ax-b)\}$. 对于$\left \| x \right \|_{1,2}$我们分行考虑次梯度. 对于行向量$x(i,1:l)(1 \leq i \leq n)$的范数$\left \| x(i,1:l) \right \|_2$, 其在$x(i,1:l)=0$处不可微, 经过计算我们可以求出在$0$是$x(i,1:l)=0$时的次梯度, 即
$$
\partial \left \| x(i,1:l) \right \|_2 =\left\{
\begin{aligned}
& \frac{x(i,1:l)}{\left \| x(i,1:l) \right \|_2} &, \left \| x(i,1:l) \right \|_2 \neq 0 \\
& 0 &, \left \| x(i,1:l) \right \|_2 = 0 \\
\end{aligned}
\right.
$$
故
$$
\partial f(x) = A^T(Ax-b) + \mu 
\begin{pmatrix}
\partial \left \| x(1,1:l) \right \|_2 \\ 
\partial \left \| x(2,1:l) \right \|_2 \\ 
\cdots \\ 
\partial \left \| x(n,1:l) \right \|_2
\end{pmatrix}
$$
对于问题(b), 我们重点考虑$\left \| x \right \|_{1,2}$我们分行考虑后的行向量范数$\left \| x(i,1:l) \right \|_2(1 \leq i \leq n)$的平滑问题. 我们引入小参数$\delta > 0$, 则$\left \| x(i,1:l) \right \|_2$可被平滑为$\sqrt{\left \| x(i,1:l) \right \|_2^2+\delta^2}-\delta$, 其中当$\delta$越小时, 平滑效果越不明显. 下图显示显示了$\delta=0.1$时的对$y=\sqrt{x^2}$的平滑效果:

![smooth](report.assets/smooth.png)

下面介绍将算法的细节.问题(a)的代码在[gl_SGD_primal.py](https://github.com/magic3007/convex-optimization/blob/main/code/gl_SGD_primal.py)中. 这里传入的默认构造参数及其定义如下:

```python
    default_opts = {
        "maxit": 2100,  			# 内循环最大迭代次数
        "thres": 1e-3,  			# 判断小量是否被认为 0 的阈值
        "step_type": "diminishing",  # 步长衰减的类型（见辅助函数）
        "alpha0": 1e-3,  			# 步长的初始值
    }
```

这里使用了连续化次梯度策略, 重定义原问题的正则化系数为$\mu_0$, 算法枚举了三个递减的正则化系数: $100\mu_0$, $10\mu_0$ 和 $\mu_0$. 外循环按照递减顺序枚举构造的正则化系数(不妨在循环内部设为$\mu$), 内循环默认运行$maxit$次迭代.

```python
    for mu in [ 100 * mu_0, 10 * mu_0, mu_0 ]:
        ......
        inn_iter = 0
        while inn_iter < maxit:
            .......
```

按照之前分析的数学形式, 其目标函数和梯度的计算如下:

```python
        def obj_func(x: np.ndarray):
            fro_term = 0.5 * np.sum((A @ x - b) ** 2)
            regular_term = np.sum(LA.norm(x, axis=1).reshape(-1, 1))
            return fro_term + mu * regular_term

        def subgrad(x: np.ndarray):
            fro_term_grad = A.T @ (A @ x - b)
            regular_term_norm = LA.norm(x, axis=1).reshape(-1, 1)
            regular_term_grad = x / ((regular_term_norm < thres) + regular_term_norm)
            grad = fro_term_grad + mu * regular_term_grad
            return grad
```

关于步长的选择, 我们仅在连续化外层循环的最后一步使用步长衰减, 其余使用固定步长:

```python
        def set_step(step_type):
            iter_hat = max(inn_iter, 1000) - 999
            if step_type == 'fixed' or mu > mu_0:
                return alpha0
            elif step_type == 'diminishing':
                return alpha0 / np.sqrt(iter_hat)
            elif step_type == 'diminishing2':
                return alpha0 / iter_hat
            else:
                logger.error("Unsupported type.")
```

在内循环内部, 我们使用次梯度法进行迭代; 同时, 对于绝对值小于给定阈值的分量, 我们直接设为0

```python
    for mu in [ 100 * mu_0, 10 * mu_0, mu_0 ]:
        ......
        inn_iter = 0
        while inn_iter < maxit:
            .......
            inn_iter += 1
            x[ np.abs(x) < thres ] = 0
            sub_g = subgrad(x)
            alpha = set_step(opts[ "step_type" ])
            x = x - alpha * sub_g
            ......
```

问题(b)的代码在[gl_GD_primal.py](https://github.com/magic3007/convex-optimization/blob/main/code/gl_GD_primal.py)中. 其默认参数为:

```python
    default_opts = {
        "maxit": 2500,  			# 最大迭代次数
        "thres": 1e-3,  			# 判断小量是否被认为 0 的阈值
        "step_type": "diminishing",  # 步长衰减的类型（见辅助函数）
        "alpha0": 1e-3,  			# 步长的初始值
        "delta": 1e-3,				# 光滑化参数
    }
```

问题(b)的代码与问题(a)的类似, 与问题(a)的主要区别在于梯度的计算上:

```python
        def subgrad(x: np.ndarray):
            fro_term_grad = A.T @ (A @ x - b)
            regular_term_grad = x / np.sqrt(np.sum(x ** 2, axis=1).reshape(-1, 1) + delta * delta)
            grad = fro_term_grad + mu * regular_term_grad
            return grad
```

以下列出这两个问题的统计数据. 在默认随机种子下, 相比于CVX mosek/gurobi, 其运行时间, 稀疏程度, 恢复效果, 迭代次数等如下. 其中运行时间约CVX-Gurobi的三倍, 最优函数值与CVX mosek/gurobi相当, 稀疏程度达到构造数据时期望的0.1, 甚至小于CVX mosek/gurobi的稀疏程度, 与CVX mosek/gurobi的恢复效果也相当接近.

| solver     | cpu  | iter | optval      | sparsity | err-to-exact | err-to-cvx-mosek | err-to-cvx-gurobi |
| ---------- | ---- | ---- | ----------- | -------- | ------------ | ---------------- | ----------------- |
| CVX-Mosek  | 0.33 | -1   | 6.10377E-01 | 0.1201   | 4.02E-05     | 0.00E+00         | 3.33E-07          |
| CVX-Gurobi | 0.71 | -1   | 6.10377E-01 | 0.1211   | 4.03E-05     | 3.33E-07         | 0.00E+00          |
| SGD Primal | 2.08 | 6300 | 6.10378E-01 | 0.0996   | 3.79E-05     | 4.30E-06         | 4.43E-06          |
| GD Primal  | 2.44 | 7500 | 6.10378E-01 | 0.0996   | 3.79E-05     | 4.31E-06         | 4.44E-06          |

SGD Primal和GD Primal的结果与ground truth $u$的比较如下. 我们可以看到, 基本上绝大部分的ground truth的分量都可以还原.

![SGD_Primal](report.assets/SGD_Primal.svg)
![GD_Primal](report.assets/GD_Primal.svg)

下图是分别是SGD Primal和GD Primal的$(f(x^k)-f^*)/f^*$随iteration变化的曲线, 其中$f^*=f(u)$. 这里垂直的线出现的原因是由于目标函数中的正则项的存在, ground truth $u$不一定最小化目标函数, 因此$f(x^k)-f^*$可能为负数. 这也从侧面说明, 我们的实现如果仅关注此目标函数下, 可以得到比原问题得到的函数值更小的解.

![relative_objective](report.assets/relative_objective.svg)

为说明算法在其他种子下的表现情况, 使用其他随机种子$seed=114514$, 其得到的结果如下: 我们可以看到算法在其他种子下表现稳定.

| solver     | cpu  | iter | optval      | sparsity | err-to-exact | err-to-cvx-mosek | err-to-cvx-gurobi |
| ---------- | ---- | ---- | ----------- | -------- | ------------ | ---------------- | ----------------- |
| CVX-Mosek  | 0.33 | -1   | 6.19068E-01 | 0.1064   | 4.03E-05     | 0.00E+00         | 8.48E-07          |
| CVX-Gurobi | 0.70 | -1   | 6.19068E-01 | 0.1064   | 4.10E-05     | 8.48E-07         | 0.00E+00          |
| SGD Primal | 2.09 | 6300 | 6.19068E-01 | 0.0996   | 3.97E-05     | 1.21E-06         | 1.84E-06          |
| GD Primal  | 2.43 | 7500 | 6.19068E-01 | 0.0996   | 3.97E-05     | 1.21E-06         | 1.84E-06          |



## Problem #3 (c) (d) & (e)

> (c) Fast (Nesterov/accelerated) gradient method for the smoothed primal problem.
> 
> (d) Proximal gradient method for the primal problem.
> 
> (e) Fast proximal gradient method for the primal problem.

**(c)**  首先讨论在光滑化后的问题上使用Nesterov梯度算法, 此部分代码在[gl_FGD_primal.py](https://github.com/magic3007/convex-optimization/blob/main/code/gl_FGD_primal.py)中. 原目标函数经过光滑化后为
$$
f(x)=\frac{1}{2}\left \| Ax-b \right\|_F^2+\mu \sum\limits_{i=1}^{n}(\sqrt{ \left\| x(i;1:l) \right\| _2^2+\delta^2} - \delta)
$$

其中$\delta > 0$ 为光滑化参数. 经过光滑化后, 整个目标函数处处光滑, 类似于Nesterov梯度算法的常见形式. 我们可以将光滑化后的优化问题写为 
$$
\min f(x)=g(x)+h(x)
$$
其中$g(x)=f(x)$是光滑的凸函数, $h(x)=0$是闭凸函数. 容易写出此时$h(x)$的近似点算子即为$ prox_{th}(x)=x$.此部分在代码中核心部分如下:

```python
        def g_func(x: np.ndarray):
            fro_term = 0.5 * np.sum((A @ x - b) ** 2)
            regular_term = np.sum(np.sqrt(np.sum(x ** 2, axis=1).reshape(-1, 1) + delta * delta) - delta)
            return fro_term + mu * regular_term

        def grad_g_func(x: np.ndarray):
            fro_term_grad = A.T @ (A @ x - b)
            regular_term_grad = x / np.sqrt(np.sum(x ** 2, axis=1).reshape(-1, 1) + delta * delta)
            return fro_term_grad + mu * regular_term_grad

      	......

        def prox_th(x: np.ndarray, t):
            """ Proximal operator of t * mu * h(x).
            """
            return x
```

不妨设选择$v^{(0)}=x^{(0)}$, 以及定义$\theta_k=\frac{2}{k+1}$, 重复以下的迭代过程:
$$
\begin{align*}
y       &= (1-\theta_k)x^{(k-1)} + \theta_kv^{(k-1)} \\
x^{(k)} &= prox_{t_k h}(y-t_k\nabla g(y)) \\
v^{(k)} &= x^{(k-1)} + \frac{1}{\theta_k}(x^{(k)} - x^{(k-1)})
\end{align*}
$$
其中$t_k$通过线搜索的方式得到. 此部分的核心代码如下:

```python
theta = 2 / (inner_iter + 1)
y = (1 - theta) * x_k + theta * v_k
grad_g_y = grad_g_func(y)

t = set_step(step_type)
x = prox_th(y - t * grad_g_y, t)
v = x_k + (x - x_k) / theta

x_k, v_k, t_k = x, v, t
```

**(d)** 线搜索部分的算法框架与核心代码如下:
$$
\begin{array}{l}
t := t_{k-1} \quad\left(\text { define } t_{0}=\hat{t}>0\right) \\
x := \operatorname{prox}_{t h}(y-t \nabla g(y)) \\
\text { while } g(x)>g(y)+\nabla g(y)^{T}(x-y)+\frac{1}{2 t}\|x-y\|_{2}^{2} \\
\qquad \begin{aligned}
t &:=\beta t \\
x &:=\operatorname{prox}_{t h}(y-t \nabla g(y))
\end{aligned}
\end{array}
$$

```python
t = t_k
g_y = g_func(y)
grad_g_y = grad_g_func(y)

def stop_condition(t):
    x = prox_th(y - t * grad_g_y, t)
    g_x = g_func(x)
    return g_x <= g_y + np.sum(grad_g_y * (x - y)) + np.sum((x - y) ** 2) / (2 * t)

for i in range(max_line_search_iter):
    if stop_condition(t):
        break
    t *= aten_coeffi
return t
```

近似点梯度算法的代码在[gl_ProxGD_primal.py](https://github.com/magic3007/convex-optimization/blob/main/code/gl_ProxGD_primal.py)中. 对于原目标函数$f(x)=\frac{1}{2}\left \| Ax-b \right\|_F^2+\mu \sum\limits_{i=1}^{n}\left\| x(i;1:l) \right\|_2$, 我们将优化问题重写为:
$$
\min f(x)=g(x)+h(x)
$$
其中$g(x)=\frac{1}{2}\left \| Ax-b \right\|_F^2$是光滑的凸函数, $h(x)=\mu \sum\limits_{i=1}^{n}\left\| x(i;1:l) \right\|_2$是强凸函数. 容易证明$p(x)=\left\| x \right\|_2$的近似点算子为:
$$
\operatorname{prox}_{t p}(x)=\left\{\begin{array}{cc}
\left(1-t /\|x\|_{2}\right) x & \|x\|_{2} \geq t \\
0 & \text { otherwise }
\end{array}\right.
$$
对于$h(x)$, 我们对$x$的每一行按照如上方式即可求得$h(x)$的近似点算子$porx_{th}(x)$, 其核心代码如下:

```python
def prox_th(x: np.ndarray, t):
    """ Proximal operator of t * mu * h(x).
    """
    t_mu = t * mu
    row_norms = LA.norm(x, axis=1).reshape(-1, 1)
    rv = x * np.clip(row_norms - t_mu, a_min=0, a_max=None) / ((row_norms < thres) + row_norms)
    return rv
```

近似点梯度法的迭代方式为:
$$
x^{(k)}=\operatorname{prox}_{t_{k} h}\left(x^{(k-1)}-t_k \nabla g\left(x^{(k-1)}\right)\right), \quad k \geq 1
$$
其中$t_k$通过线搜索的方式得到, 其算法框架与核心代码为:
$$
\begin{array}{l}
\text { define } G_{t}(x)=\frac{1}{t}\left(x-\operatorname{prox}_{t h}(x-t \nabla g(x))\right)\\
t := \hat{t} > 0 \\
\text { while } g(x-tG_t(x))>g(x) -t\nabla g(x)^{T}G_t(x)+\frac{t}{2}\|G_t(x)\|_{2}^{2} \\
\qquad \begin{aligned}
t &:=\beta t \\
\end{aligned}
\end{array}
$$

```python
 g_x = g(x)

def stop_condition(x, t):
    gt_x = Gt(x, t)
    return (g(x - t * gt_x)
            <= g_x - t * np.sum(grad_g * gt_x) + 0.5 * t * np.sum(gt_x ** 2))

alpha = alpha0
for i in range(max_line_search_iter):
    if stop_condition(x, alpha):
        break
    alpha *= aten_coeffi
return alpha
```

```
g_x = g(x)

def stop_condition(x, t):
    gt_x = Gt(x, t)
    return (g(x - t * gt_x)
            <= g_x - t * np.sum(grad_g * gt_x) + 0.5 * t * np.sum(gt_x ** 2))

alpha = alpha0
for i in range(max_line_search_iter):
    if stop_condition(x, alpha):
        break
    alpha *= aten_coeffi
return alpha
```

**(e)** 在原问题上使用Nesterov梯度算法见[gl_FProxGD_primal.py](https://github.com/magic3007/convex-optimization/blob/main/code/gl_FProxGD_primal.py).基本与问题(c)类似, 我们仅仅需要重新定义优化问题:
$$
\min f(x)=g(x)+h(x)
$$
其中$g(x)=\frac{1}{2}\left \| Ax-b \right\|_F^2$是光滑的凸函数, $h(x)=\mu \sum\limits_{i=1}^{n}\left\| x(i;1:l) \right\|_2$是强凸函数, 其与问题(c)主要不同的代码如下:

```python
def g_func(x: np.ndarray):
    return 0.5 * np.sum((A @ x - b) ** 2)

def grad_g_func(x: np.ndarray):
    return A.T @ (A @ x - b)

......

def prox_th(x: np.ndarray, t):
    """ Proximal operator of t * mu * h(x).
    """
    t_mu = t * mu
    row_norms = LA.norm(x, axis=1).reshape(-1, 1)
    rv = x * np.clip(row_norms - t_mu, a_min=0, a_max=None) / ((row_norms < thres) + row_norms)
    return rv
```

以下列出了问题(a)-(e)得到的图表和统计数据. 在默认随机种子下, 相比于CVX mosek/gurobi, 其运行时间, 稀疏程度, 恢复效果, 迭代次数等如下. 我们可以看到在原问题上使用近似点梯度算法和Nesterov梯度算法具有较小的迭代数和较小的运行时间, 从统计数据上看基本达到和次梯度法相近的解, 但运行时间要小得多.

![relative_objective_a_e](report.assets/relative_objective_a_e.svg)

| solver         | cpu  | iter | optval      | sparsity | err-to-exact | err-to-cvx-mosek | err-to-cvx-gurobi |
| -------------- | ---- | ---- | ----------- | -------- | ------------ | ---------------- | ----------------- |
| CVX-Mosek      | 0.32 | -1   | 6.10377E-01 | 0.1201   | 4.02E-05     | 0.00E+00         | 3.33E-07          |
| CVX-Gurobi     | 0.73 | -1   | 6.10377E-01 | 0.1211   | 4.03E-05     | 3.33E-07         | 0.00E+00          |
| SGD Primal     | 2.02 | 6300 | 6.10378E-01 | 0.0996   | 3.79E-05     | 4.30E-06         | 4.43E-06          |
| GD Primal      | 2.41 | 7500 | 6.10378E-01 | 0.0996   | 3.79E-05     | 4.31E-06         | 4.44E-06          |
| FGD Primal     | 1.24 | 2037 | 6.10378E-01 | 0.1221   | 4.21E-05     | 2.39E-06         | 2.27E-06          |
| ProxGD Primal  | 1.53 | 1768 | 6.10377E-01 | 0.0996   | 3.79E-05     | 4.38E-06         | 4.52E-06          |
| FProxGD Primal | 1.09 | 1721 | 6.10377E-01 | 0.0996   | 3.79E-05     | 4.38E-06         | 4.52E-06          |

## Problem #4 (f)  (g) & (h)

> (f) Augmented Lagrangian method for the dual problem.
>
> (g) Alternating direction method of multipliers for the dual problem.
> 
> (h) Alternating direction method of multipliers with linearization for the primal problem.

首先引入约束, 构造对偶问题. 不妨设$y=Ax-b$, 原问题写成
$$
\begin{aligned}
\min_{x,y} & f(x)+g(y) \\
\text{s.t.} &\ Ax-b-y=0
\end{aligned}
$$
其中$f(x)=\lVert x \rVert_{1,2}$, $g(y)=\frac{1}{2}\lVert y \rVert_F^2$, 其拉格朗日函数为
$$
L(x,y,z) = f(x)+g(y)+\langle z,Ax-b-y \rangle
$$

$$
h(z)= \inf_{x,y} L(x,y,z) = - f^*(-A^Tz) - g^*(z) - \langle b,z\rangle
$$

设$p(x)=\lVert x \rVert_2(x \in R^l)$为向量的2范数, 其对偶范数为其本身, 故其共轭函数为$p^*(x)=\left \{ \begin{aligned} 0,&\ \lVert x \rVert_2 \leq 1 \\ \infty,&\ otherwise \end{aligned} \right.$. 由于$f(x)=\sum\limits_{i=1}^{n}p(x_i)$, 故$f^*(x)=\sum\limits_{i=1}^{n}p^*(x_i)$. 另一方面, 容易得到$g$的共轭函数即为自身, 即$g^*(z)=g(z)$. 故其对偶问题为:
$$
\begin{aligned}
\min_{z,u} &\  g(z)+\langle b,z\rangle \\
\text{s.t.} &\ u+A^Tz=0 \\
&\ 1 - \lVert u_i \rVert_2\geq 0, i=1,\ldots, n
\end{aligned}
$$
引入辅助变量$v_i = \lVert u_i \rVert_2 - 1$, 得到:
$$
\begin{aligned}
\min_{z,u,v} &\  g(z)+\langle b,z\rangle \\
\text{s.t.}\ &u+A^Tz &= 0 \\
\ &1-\lVert u_i \rVert_2 - v_i &= 0 \\
\ &v_i &\geq 0
\end{aligned}
$$
其增广拉格朗日函数为:
$$
\begin{aligned}
L_t(z, u,v, \lambda, \omega)&=g(z)+\langle b,z \rangle - \langle \lambda, u+A^Tz \rangle - \sum \omega_i(1-\lVert u_i \rVert_2 - v_i) + \frac{t}{2}\lVert u+A^Tz\rVert_F^2 + \frac{t}{2}\sum(1-\lVert u_i \rVert_2 - v_i)^2\\
\text{s.t.}\ v_i &\geq 0
\end{aligned}
$$


根据一阶条件$\frac{\partial L_t(z, u,v, \lambda, \omega)}{\partial v}=0$可得到$v_i=\max(1-\lVert u_i \rVert_2-\frac{w_i}{t},0)$.  故可消去$v$.
$$
L_t(z,u,\lambda, \omega)=g(z)+\langle b,z \rangle - \langle \lambda, u+A^Tz \rangle + \frac{t}{2}\lVert u+A^Tz\rVert_F^2  + \sum \phi(1-\lVert u_i \rVert_2, w_i, t)
$$
其中$\phi(1-\lVert u_i \rVert_2, w_i, t)=\left \{ \begin{aligned} -\omega_i(1-\lVert u_i \rVert_2)+\frac{t}{2}(1-\lVert u_i \rVert_2)^2,&\ 1-\lVert u_i \rVert_2-\frac{w_i}{t} \leq 0	 \\ -\frac{w_i^2}{2t},&\ otherwise \end{aligned} \right.$

更新方式为:
$$
\begin{aligned}
z^{k+1}, u^{k+1} &= argmin_{z,u} L_t(z,u,\lambda^{k}, \lambda^{w}) \\
\lambda^{k+1} &= \lambda^k - \frac{t}{2}(u^{k+1}+A^Tz^{k+1}) \\
\omega^{k+1} &= \max(\omega^{k}-t(1-\lVert u^{k+1}_i \rVert_2), 0)
\end{aligned}
$$
