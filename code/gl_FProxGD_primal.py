import numpy as np
from numpy import linalg as LA
import logging
from utils.stopwatch import Stopwatch

logger = logging.getLogger("opt")


def gl_FProxGD_primal(x0: np.ndarray, A: np.ndarray, b: np.ndarray, mu_0, opts: dict):
    default_opts = {
        "maxit": 1500,  # 最大迭代次数
        "thres": 1e-3,  # 判断小量是否被认为 0 的阈值
        "step_type": "line_search",  # 步长衰减的类型（见辅助函数）
        "alpha0": 1e-3,  # 步长的初始值
        "ftol": 1e-6,  # 停机准则，当目标函数历史最优值的变化小于该值时认为满足
        "stable_len_threshold": 70,
        "line_search_attenuation_coeffi": 0.98,
        "maxit_line_search_iter": 5,
    }
    # The second dictionary's values overwrite those from the first.
    opts = {**default_opts, **opts}
    sparsity_func = lambda x: np.sum(np.abs(x) > 1e-6 * np.max(np.abs(x))) / x.size

    def real_obj_func(x: np.ndarray):
        fro_term = 0.5 * np.sum((A @ x - b) ** 2)
        regular_term = np.sum(LA.norm(x, axis=1).reshape(-1, 1))
        return fro_term + mu_0 * regular_term

    out = {
        "fvec": None,  # 每一步迭代的 LASSO 问题目标函数值
        "grad_hist": None,  # 可微部分梯度范数的历史值
        "f_hist": None,  # 目标函数的历史值
        "f_hist_best": None,  # 目标函数每一步迭代对应的历史最优值
        "tt": None,  # 运行时间
        "flag": None  # 标记是否收敛
    }

    maxit, ftol, alpha0 = opts["maxit"], opts["ftol"], opts["alpha0"]
    stable_len_threshold = opts["stable_len_threshold"]
    thres = opts["thres"]
    step_type = opts['step_type']
    aten_coeffi = opts['line_search_attenuation_coeffi']
    max_line_search_iter = opts['maxit_line_search_iter']

    logger.debug("alpha0= {:10E}".format(alpha0))
    f_hist, f_hist_best, sparsity_hist = [], [], []
    v_hist, t_hist = [], []
    f_best = np.inf

    x_k = np.copy(x0)

    stopwatch = Stopwatch()
    stopwatch.start()
    k = 0
    for mu in [100 * mu_0, 10 * mu_0, mu_0]:
        logger.debug("new mu= {:10E}".format(mu))

        # min f(x) = g(x) + h(x)
        # g(x) = 0.5 * |Ax-b|_F^2
        # h(x) = mu * |x|_{1,2}

        def g_func(x: np.ndarray):
            return 0.5 * np.sum((A @ x - b) ** 2)

        def grad_g_func(x: np.ndarray):
            return A.T @ (A @ x - b)

        v_k = np.copy(x_k)
        t_k = alpha0

        def prox_th(x: np.ndarray, t):
            """ Proximal operator of t * mu * h(x).
            """
            t_mu = t * mu
            row_norms = LA.norm(x, axis=1).reshape(-1, 1)
            rv = x * np.clip(row_norms - t_mu, a_min=0, a_max=None) / ((row_norms < thres) + row_norms)
            return rv

        inner_iter = 0

        def set_step(step_type: str):
            iter_hat = max(inner_iter, 1000) - 999
            if step_type == 'fixed':
                return alpha0
            elif step_type == 'diminishing':
                return alpha0 / np.sqrt(iter_hat)
            elif step_type == 'diminishing2':
                return alpha0 / iter_hat
            elif step_type == 'line_search':

                t = t_k
                g_y = g_func(y)

                def stop_condition(t):
                    x = prox_th(y - t * grad_g_y, t)
                    g_x = g_func(x)
                    return g_x <= g_y + np.sum(grad_g_y * (x - y)) + np.sum((x - y) ** 2) / (2 * t)

                for i in range(max_line_search_iter):
                    if stop_condition(t):
                        break
                    t *= aten_coeffi
                return t

            else:
                logger.error("Unsupported type.")

        stable_len = 0

        while inner_iter < maxit:
            # Record current objective value
            f_now = real_obj_func(x_k)
            f_hist.append(f_now)

            f_best = min(f_best, f_now)
            f_hist_best.append(f_best)

            sparsity_hist.append(sparsity_func(x_k))

            v_hist.append(v_k)

            t_hist.append(t_k)

            k += 1
            inner_iter += 1

            if (k > 1
                    and abs(f_hist[k - 1] - f_hist[k - 2]) / abs(f_hist[k - 2]) < ftol
                    and abs(sparsity_hist[k - 1] - sparsity_hist[k - 2]) / abs(sparsity_hist[k - 2]) < ftol):
                stable_len += 1
            else:
                stable_len = 0
            if stable_len > stable_len_threshold:
                break

            x_k[np.abs(x_k) < thres] = 0

            theta = 2 / (inner_iter + 1)
            y = (1 - theta) * x_k + theta * v_k
            grad_g_y = grad_g_func(y)

            t = set_step(step_type)
            # logger.debug("t: {}".format(t))
            x = prox_th(y - t * grad_g_y, t)
            v = x_k + (x - x_k) / theta

            x_k, v_k, t_k = x, v, t

            if k % 100 == 0:
                logger.debug(
                    'iter= {:5}, objective= {:10E}, sparsity= {:3f}'.format(k, f_now.item(), sparsity_func(x)))

    elapsed_time = stopwatch.elapsed(time_format=Stopwatch.TimeFormat.kMicroSecond) / 1e6
    out = {
        "tt": elapsed_time,
        "fval": real_obj_func(x),
        "f_hist": f_hist,
        "f_hist_best": f_hist_best
    }

    return x, k, out
