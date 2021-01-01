import numpy as np
from numpy import linalg as LA
import logging
from utils.stopwatch import Stopwatch
import math

logger = logging.getLogger("opt")


def gl_Admm_dual(x0: np.ndarray, A: np.ndarray, b: np.ndarray, mu, opts: dict):
    default_opts = {
        "maxit": 100,  # 最大迭代次数
        "thres": 1e-3,  # 判断小量是否被认为 0 的阈值
        "tau": (1 + math.sqrt(5)) * 0.5,
        "rho": 1e2,
        "converge_len": 20,
    }

    # The second dictionary's values overwrite those from the first.
    opts = {**default_opts, **opts}

    def sparsity_func(x: np.ndarray):
        return np.sum(np.abs(x) > 1e-6 * np.max(np.abs(x))) / x.size

    def real_obj_func(x: np.ndarray):
        fro_term = 0.5 * np.sum((A @ x - b) ** 2)
        regular_term = np.sum(LA.norm(x, axis=1).reshape(-1, 1))
        return fro_term + mu * regular_term

    out = {
        "fvec": None,  # 每一步迭代的 LASSO 问题目标函数值
        "f_hist": None,  # 目标函数的历史值
        "f_hist_best": None,  # 目标函数每一步迭代对应的历史最优值
        "tt": None,  # 运行时间
    }

    maxit, thres = opts["maxit"], opts["thres"]
    rho, tau = opts['rho'], opts['tau']
    converge_len = opts['converge_len']

    f_hist, f_hist_best, sparsity_hist = [], [], []
    f_best = np.inf

    def projection_functor(x: np.array):
        row_norms = LA.norm(x, axis=1, ord=2).reshape(-1, 1)
        return mu * x / np.clip(row_norms, a_min=mu, a_max=None)

    x_k = np.copy(x0)
    z_k = np.zeros_like(b)
    u_k = np.zeros_like(x_k)

    stopwatch = Stopwatch()
    stopwatch.start()

    k = 0

    L = LA.cholesky(np.identity(A.shape[0]) + rho * A @ A.T)

    length = 0

    while k < maxit:
        k += 1
        z = LA.solve(L.T, LA.solve(L, A @ (x_k - rho * u_k) - b))
        u = projection_functor(x_k / rho - A.T @ z)
        x = x_k - tau * rho * (u + A.T @ z)

        r_k = u + A.T @ z  # 原始可行性
        s_k = A @ (u_k - u)  # 对偶可行性

        z_k, u_k, x_k = z, u, x

        f_now = real_obj_func(x_k)
        f_hist.append(f_now)

        f_best = min(f_best, f_now)
        f_hist_best.append(f_best)

        sparsity_now = sparsity_func(x_k)

        sparsity_hist.append(sparsity_now)

        if k % 1 == 0:
            logger.debug('iter= {:5}, objective= {:10E}, sparsity= {:3f}'.format(k, f_now.item(), sparsity_now.item()))

        r_k_norm = LA.norm(r_k, ord=2)
        s_k_norm = LA.norm(s_k, ord=2)
        if r_k_norm < thres and s_k_norm < thres:
            length += 1
        else:
            length = 0

        if length >= converge_len:
            break

    elapsed_time = stopwatch.elapsed(time_format=Stopwatch.TimeFormat.kMicroSecond) / 1e6
    out = {
        "tt": elapsed_time,
        "fval": real_obj_func(x_k),
        "f_hist": f_hist,
        "f_hist_best": f_hist_best
    }

    return x_k, k, out
