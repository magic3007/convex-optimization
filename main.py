import numpy as np
from numpy import random
from numpy.linalg import norm
import argparse
import logging
import os
from matplotlib import pyplot as plt


# min 0.5 * ||A * x - b||_2^2 + mu * ||x||_{1,2}

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
    formatter = logging.Formatter('%(asctime)s: %(levelname)-8s %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='a')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler( )
    streamHandler.setFormatter(formatter)
    log_setup.setLevel(level)
    log_setup.addHandler(fileHandler)
    log_setup.addHandler(streamHandler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A demo that solves the optimization problem '
                                                 '$min 0.5 * ||A * x - b||_2^2 + mu * ||x||_{1,2}$')
    parser.add_argument('--log', type=str, default='web.log', help='Path to the logging file.')
    parser.add_argument('--dest_dir', type=str, default='figures', help='Destination directory.')
    args = parser.parse_args( )

    log_file_path = args.log
    destination_directory = args.dest_dir

    setup_logger("opt", log_file_path)
    logger = logging.getLogger('opt')
    logger.info("========================== New Log ========================================")

    if os.path.isdir(destination_directory) is False:
        os.makedirs(destination_directory)
        logger.info("Create directory: %s" % destination_directory)

    n, m, l, mu, A, b, u, x0, errfun, errfun_exact, sparsity = gen_data( )

    fig = plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(0, n, 1), u[ :, 0 ], '*')
    plt.plot(np.arange(0, n, 1), u[ :, 1 ], 'o')
    plt.xlim(0, n)
    plt.title(r'(1) exact solution $u$')
    plt.show()
    plt.savefig(os.path.join(destination_directory, 'ground_truth.svg'))
