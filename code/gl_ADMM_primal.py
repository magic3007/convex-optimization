import numpy as np
from numpy import linalg as LA
import logging
from utils.stopwatch import Stopwatch
import math

logger = logging.getLogger("opt")


def gl_Admm_primal(x0: np.ndarray, A: np.ndarray, b: np.ndarray, mu, opts: dict):
    default_opts = {
        "maxit": 100,  # 最大迭代次数
        "thres": 1e-3,  # 判断小量是否被认为 0 的阈值
        "tau": (1 + math.sqrt(5)) * 0.5,
        "rho": 1e-2,
        "eta_0": 100,
        "converge_len": 10,
        "converge_thres": 1e-5,
        "step_type": "fixed",
    }

    # The second dictionary's values overwrite those from the first.
    opts = {**default_opts, **opts}
    sparsity_func = lambda x: np.sum(np.abs(x) > 1e-6 * np.max(np.abs(x))) / x.size

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
    rho, tau, eta_0 = opts['rho'], opts['tau'], opts['eta_0']
    converge_len = opts['converge_len']
    converge_thres = opts['converge_thres']
    step_type = opts['step_type']

    f_hist, f_hist_best, sparsity_hist = [], [], []
    f_best = np.inf

    def prox_tf(x: np.array, t):
        t_mu = t * mu
        row_norms = LA.norm(x, axis=1).reshape(-1, 1)
        rv = x * np.clip(row_norms - t_mu, a_min=0, a_max=None) / ((row_norms < thres) + row_norms)
        return rv

    x_k = np.copy(x0)
    y_k = x_k
    z_k = x_k

    stopwatch = Stopwatch()
    stopwatch.start()

    k = 0

    L = LA.cholesky(rho * np.identity(A.shape[1]) + A.T @ A)
    AT_b = A.T @ b

    length = 0

    def set_step(step_type: str):
        if step_type == 'fixed':
            return eta_0
        elif step_type == 'diminishing':
            return eta_0 / np.sqrt(k)
        elif step_type == 'diminishing2':
            return eta_0 / k

    while k < maxit:
        k += 1
        eta = set_step(step_type)
        y = LA.solve(L.T, LA.solve(L, AT_b - z_k + rho * x_k))
        # x = prox_tf(y + z_k / rho, 1/rho)
        x = prox_tf(x_k - eta * rho * (x_k - y - z_k/rho), eta)
        z = z_k - tau * rho * (x - y)

        r_k = x - y     # 原始可行性
        s_k = y - y_k   # 对偶可行性

        x_k, y_k, z_k = x, y, z

        f_now = real_obj_func(x_k)
        f_hist.append(f_now)

        f_best = min(f_best, f_now)
        f_hist_best.append(f_best)

        sparsity_hist.append(sparsity_func(x_k))

        if k % 1 == 0:
            logger.debug('iter= {:5}, objective= {:10E}, sparsity= {:3f}'.format(k, f_now.item(), sparsity_func(x_k)))

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
