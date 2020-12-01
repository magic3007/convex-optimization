import numpy as np
from numpy import linalg as LA
import logging
from utils.stopwatch import Stopwatch

logger = logging.getLogger("opt")


def gl_SGD_primal(x0: np.ndarray, A: np.ndarray, b: np.ndarray, mu_0, opts: dict):
    default_opts = {
        "maxit": 20000,  # 最大迭代次数
        "thres": 1e-3,  # 判断小量是否被认为为 0 的阈值
        "step_type": "diminishing",  # 步长衰减的类型（见辅助函数）
        "alpha0": 1e-3,  # 步长的初始值
        "ftol": 1e-5,  # 停机准则，当目标函数历史最优值的变化小于该值时认为满足
        "stable_len_threshold": 100,
        "continuous_subgradient_flag": False
    }
    # The second dictionary's values overwrite those from the first.
    opts = {**default_opts, **opts}

    out = {
        "fvec": None,  # 每一步迭代的 LASSO 问题目标函数值
        "grad_hist": None,  # 可微部分梯度范数的历史值
        "f_hist": None,  # 目标函数的历史值
        "f_hist_best": None,  # 目标函数每一步迭代对应的历史最优值
        "tt": None,  # 运行时间
        "flag": None  # 标记是否收敛
    }

    maxit, ftol, alpha0 = opts[ "maxit" ], opts[ "ftol" ], opts[ "alpha0" ]
    stable_len_threshold = opts[ "stable_len_threshold" ]
    thres = opts[ "thres" ]

    if opts[ "continuous_subgradient_flag" ] is True:
        L = np.max(LA.eigvals(A.T @ A))
        alpha0 = 1. / L.real

    logger.debug("alpha0= {:10E}".format(alpha0))
    f_hist, f_hist_best = [ ], [ ]
    f_best = np.inf

    x = np.copy(x0)
    stopwatch = Stopwatch( )
    stopwatch.start( )
    k = 0
    stable_len = 0
    for mu in [100*mu_0, 50*mu_0, 25*mu_0, 10*mu_0, mu_0]:
        logger.debug("new mu= {:10E}".format(mu))

        def obj_func(x: np.ndarray):
            temp = A @ x - b
            fro_term = 0.5 * np.sum(temp ** 2)
            regular_term = np.sum(LA.norm(x, axis=1).reshape(-1, 1))
            return fro_term + mu * regular_term

        def subgrad(x: np.ndarray):
            fro_term_grad = A.T @ (A @ x - b)
            regular_term_norm = LA.norm(x, axis=1).reshape(-1, 1)
            regular_term_grad = x / ((regular_term_norm < thres) + regular_term_norm)
            grad = fro_term_grad + mu * regular_term_grad
            return grad

        inner_iter = 0

        def set_step(step_type):
            iter_hat = max(inner_iter, 1000) - 999
            if step_type == 'fixed':
                return alpha0
            elif step_type == 'diminishing':
                return alpha0 / np.sqrt(iter_hat)
            elif step_type == 'diminishing2':
                return alpha0 / iter_hat
            else:
                logger.error("Unsupported type.")

        while inner_iter < maxit:
            # Record current objective value
            f_now = obj_func(x)
            f_hist.append(f_now)

            f_best = min(f_best, f_now)
            f_hist_best.append(f_best)
            k += 1
            inner_iter += 1

            if k > 1 and abs(f_hist[ k - 1 ] - f_hist[ k - 2 ]) / abs(f_hist[ k - 2 ]) < ftol:
                stable_len += 1
            else:
                stable_len = 0
            # if stable_len > stable_len_threshold:
            #     break

            x[ np.abs(x) < thres ] = 0
            sub_g = subgrad(x)
            alpha = set_step(opts[ "step_type" ])
            x = x - alpha * sub_g

            if k % 100 == 0:
                logger.debug('iter= {:5}, objective= {:10E}'.format(k, f_now.item( )))

    elapsed_time = stopwatch.elapsed(time_format=Stopwatch.TimeFormat.kMicroSecond) / 1e6
    out = {
        "tt": elapsed_time,
        "fval": obj_func(x),
        "f_hist": f_hist,
        "f_hist_best": f_hist_best
    }

    return x, k + 1, out
