import numpy as np
from numpy import random
from numpy.linalg import norm
import numpy.linalg as LA
import argparse
import logging
import os
from matplotlib import pyplot as plt
import cvxpy as cp
from gl_cvx_gurobi import gl_cvx_gurobi
from gl_cvx_mosek import gl_cvx_mosek
from gl_gurobi import gl_gurobi
from gl_mosek import gl_mosek
from gl_SGD_primal import gl_SGD_primal
from gl_GD_primal import gl_GD_primal


# min 0.5 * ||A * x - b||_2^2 + mu * ||x||_{1,2}
# Credit to
#   http://niaohe.ise.illinois.edu/IE598_2016/lasso_demo/index.html

def obj_func(x: np.ndarray):
    temp = np.matmul(A, x) - b
    fro_term = 0.5 * np.sum(np.multiply(temp, temp))
    regular_term = np.sum(np.apply_along_axis(func1d=lambda row: LA.norm(row), axis=1, arr=x))
    return fro_term + mu * regular_term

def gen_data():
    seed = 97006855
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


def setup_logger(logger_name, log_file, level=logging.INFO):
    log_setup = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s: %(levelname)-5s %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='a')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler( )
    streamHandler.setFormatter(formatter)
    log_setup.setLevel(level)
    log_setup.addHandler(fileHandler)
    log_setup.addHandler(streamHandler)


def plot_result(mode: str, file_name: str, ground_truth, cvx_mosek_rv, cvx_gurobi_rv, x):
    plt.clf()
    fig = plt.figure(1)

    plt.subplot(2, 1, 1)
    plt.plot(np.arange(0, n, 1), ground_truth[ :, 0 ], 'r*', label='ground truth')
    plt.plot(np.arange(0, n, 1), cvx_mosek_rv[ :, 0 ], 'g^', label='CVX-Mosek')
    plt.plot(np.arange(0, n, 1), cvx_gurobi_rv[ :, 0 ], 'bv', label='CVX-Gurobi')
    plt.plot(np.arange(0, n, 1), x[ :, 0 ], 'mo', label=mode)
    plt.xlim(0, n)
    plt.title('Results on the 1st dimension')

    plt.subplot(2, 1, 2)
    plt.plot(np.arange(0, n, 1), ground_truth[ :, 1 ], 'r*', label='ground truth')
    plt.plot(np.arange(0, n, 1), cvx_mosek_rv[ :, 1 ], 'g^', label='CVX-Mosek')
    plt.plot(np.arange(0, n, 1), cvx_gurobi_rv[ :, 1 ], 'bv', label='CVX-Gurobi')
    plt.plot(np.arange(0, n, 1), x[ :, 1 ], 'mo', label=mode)
    plt.xlim(0, n)
    plt.title('Results on the 2nd dimension')

    plt.show( )
    plt.savefig(file_name)


cvx_mosek_rv, cvx_gurobi_rv = None, None


def solve_routine(mode: str, func, x0, A, b, mu, opts, u, errfun, errfun_exact, sparsity):
    x, num_iters, out = func(x0, A, b, mu, opts)
    solve_time = out[ "tt" ]
    fval = out[ "fval" ]
    log_dict = {
        "cpu: %5.2f": solve_time,
        "iter: %5d": -1 if num_iters is None else num_iters,
        "optval: %6.5E": fval,
        "sparsity: %4.3f": sparsity(x),
        "err-to-exact: %3.2E": errfun_exact(x),
        "err-to-cvx-mosek: %3.2E": errfun(cvx_mosek_rv, x),
        "err-to-cvx-gurobi: %3.2E": errfun(cvx_gurobi_rv, x)
    }
    log_fmt = "[%-10s]: " + ', '.join(log_dict.keys( ))
    log_fags = tuple([ mode ] + list(log_dict.values( )))
    logger = logging.getLogger('opt')
    logger.info(log_fmt % log_fags)
    # plot_result(mode, os.path.join(destination_directory, "%s.svg" % mode), u, cvx_mosek_rv, cvx_gurobi_rv, x)
    return x, num_iters, out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A demo that solves the optimization problem '
                                                 '$\min_x{0.5 * ||A * x - b||_2^2 + mu * ||x||_{1,2}}$')
    parser.add_argument('--log', type=str, default='opt.log', help='Path to the logging file.')
    parser.add_argument('--dest_dir', type=str, default='figures', help='Destination directory.')
    args = parser.parse_args( )

    log_file_path = args.log
    destination_directory = args.dest_dir

    setup_logger("opt", log_file_path, level=logging.INFO)
    logger = logging.getLogger('opt')
    logger.info("========================== New Log ========================================")

    if os.path.isdir(destination_directory) is False:
        os.makedirs(destination_directory)
        logger.info("Create directory: %s" % destination_directory)

    installed_solvers = cp.installed_solvers( )
    logger.info("Installed solvers for cvxpy: " + str(installed_solvers))

    n, m, l, mu, A, b, u, x0, errfun, errfun_exact, sparsity = gen_data( )

    fig = plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(0, n, 1), u[ :, 0 ], '*')
    plt.plot(np.arange(0, n, 1), u[ :, 1 ], 'o')
    plt.xlim(0, n)
    plt.title(r'(1) exact solution $u$')
    # plt.show( )
    plt.savefig(os.path.join(destination_directory, 'ground_truth.svg'))

    cvx_mosek_rv, _, _ = gl_cvx_mosek(x0, A, b, mu, {})
    cvx_gurobi_rv, _, _ = gl_cvx_gurobi(x0, A, b, mu, {})
    solvers = {
        'CVX-Mosek': gl_cvx_mosek,
        'CVX-Gurobi': gl_cvx_gurobi,
        # 'Mosek': gl_mosek,
        # 'Gurobi': gl_gurobi,
        'SGD Primal': gl_SGD_primal,
        # "GD Primal": gl_GD_primal,
    }

    f_hists = {}
    for mode, solver in solvers.items( ):
        _, _, out = solve_routine(mode, solver, x0, A, b, mu, {}, u, errfun, errfun_exact, sparsity)
        if 'f_hist' in out:
            f_hists[mode] = out["f_hist"]

    file_name = 'relative_error.svg'
    fig, ax = plt.subplots(figsize=(9, 6))
    plot_solver_color = {
        'SGD Primal': 'g',
        "GD Primal": 'b',
    }
    f_star = obj_func(u)
    for mode, f_hist in f_hists.items():
        f_hist = (f_hist - f_star) / f_star
        plt.semilogy(np.arange(0, len(f_hist)), np.array(f_hist), plot_solver_color[mode], linewidth=2, label=mode)
    plt.legend(prop={'size': 12})
    plt.ylabel('$(f(x^k)-f^*)/f^*$')
    plt.xlabel('Iteration')
    plt.savefig(file_name)
