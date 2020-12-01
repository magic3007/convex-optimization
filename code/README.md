## Prerequisites

- Python >= 3.7

- The required Python packages are listed in [`requirement.txt`](https://github.com/magic3007/convex-optimization/blob/main/requirements.txt). You can install all these packages by running the command `pip install -r requirements.txt` in your shell. 

- However, since `Mosek` and `gurobipy` are just the python interfaces for commercial solvers `Mosek` and `Gurobi` respectively and `cvxpy` also relies on these commercial solvers. Therefore, just installing these packages without using these solvers as backends is not enough. To install these solvers, please refer to the next section.

  ```
  numpy~=1.19.4
  cvxpy~=1.1.7
  matplotlib~=3.3.3
  Mosek~=9.2.29
  gurobipy~=9.1.0
  ```

## Install

- Install Python Package: `pip install -r requirement.txt`
- Install solver `Mosek` and check that python package `mosek` was properly installed: please refer to [here](https://docs.mosek.com/9.2/pythonapi/install-interface.html).
- Install solver `Gurobi` and check that python package `gurobipy` was properly installed: please refer to [this blog](http://www.matthewdgilbert.com/blog/introduction-to-gurobipy.html).
- Add `Mosek` and `Gurobi` support for python package `cvxpy`: please refer to [here](https://www.cvxpy.org/install/index.html).

## Usage

```bash
$ python main.py -h
usage: main.py [-h] [--log LOG] [--dest_dir DEST_DIR]

A demo that solves the optimization problem $\min_x{0.5 * ||A * x - b||_2^2 + mu * ||x||_{1,2}}$

optional arguments:
  -h, --help           show this help message and exit
  --log LOG            Path to the logging file. (default: opt.log)
  --dest_dir DEST_DIR  Destination directory. (default: figures)
```
