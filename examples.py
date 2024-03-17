import numpy as np
from dataclasses import dataclass
from typing import Callable
import scipy.sparse as ssp

@dataclass
class Problem:
    f: Callable[[np.ndarray], np.ndarray]
    G: Callable[[np.ndarray], np.ndarray]
    omega: Callable[[np.ndarray], list[np.ndarray]]
    proj: Callable[[np.ndarray], np.ndarray]
    z0: np.ndarray

def p1_proj(x):
    y = np.copy(x)
    y[1] = np.maximum(0, y[1])
    return y

p1 = Problem(
    lambda x: np.array([
        x[1]**2,
        x[1] * (1 + x[0]**2)
    ]),
    lambda x: np.array([
        [0, 2*x[1]],
        [2 * x[0] * x[1], 1 + x[0]**2]
    ]),
    lambda x: [x[1]>=0],
    lambda x: p1_proj(x),
    np.array([1, 0.5])
)

def p2_proj(x):
    y = np.copy(x)
    y[0:2] = np.clip(y[0:2], -50, 50)
    y[2:] = np.clip(y[2:], 0, None)
    return y

p2 = Problem(
    lambda x: np.array([
        -x[0] - x[1] + 1 + x[2],
        -x[0]**2 - x[1]**2 + 1 + x[3],
        -9*x[0]**2 - x[1]**2 + 9 + x[4],
        -x[0]**2 + x[1] + x[5],
        -x[1]**2 + x[0] + x[6]
    ]),
    lambda x: np.array([
        [-1, -1, 1, 0, 0, 0, 0],
        [-2*x[0], -2*x[1], 0, 1, 0, 0, 0],
        [-18*x[0], -2*x[1], 0, 0, 1, 0, 0],
        [-2*x[0], 1, 0, 0, 0, 1, 0],
        [1, -2*x[1], 0, 0, 0, 0, 1]
    ]),
    lambda x: [x[0:2]>=-50, x[0:2]<=50, x[2:]>=0],
    lambda x: p2_proj(x),
    np.array([1, 1, 1.5, 0, 0, 0, 0])
)

_min_prime = lambda a, b: 1. if b-a>0 else 0.

p3 = Problem(
    lambda x: np.array([
        x[0]*x[1] - x[2],
        x[0]**2 + x[1] - 1 - x[3],
        min(x[0], x[2]),
        min(x[1], x[3])
    ]),
    lambda x: np.array([
        [x[1], x[0], -1, 0],
        [2*x[0], 1, 0, -1],
        [_min_prime(x[0],x[2]), 0, _min_prime(x[2],x[0]), 0],
        [0, _min_prime(x[1], x[3]), 0, _min_prime(x[3], x[1])]
    ]),
    lambda x: [x>=0],
    lambda x: np.maximum(x, 0),
    np.array([2, 1, 0, 0])
)

def p4_proj(x):
    y = np.copy(x)
    y[4:] = np.clip(y[4:], 0, None)
    return y

p4 = Problem(
    lambda x: np.array([
        x[3] + x[4] - x[5] - x[8],
        x[3] + x[1] + x[2] - x[6] - x[8],
        x[1] + x[2] - x[8],
        x[0] + x[1] - x[7],
        x[0] + x[9],
        -x[0] + x[10],
        1 - x[1] + x[11],
        -x[3] + x[12],
        -x[0] - x[1] - x[2] + x[13],
        min(x[4], x[9]),
        min(x[5], x[10]),
        min(x[6], x[11]),
        min(x[7], x[12]),
        min(x[8], x[13])
    ]),
    lambda x: np.array([
        [0, 0, 0, 1, 1, -1, 0, 0, -1, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, -1, 0, -1, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [-1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, _min_prime(x[4],x[9]), 0, 0, 0, 0, _min_prime(x[9],x[4]), 0, 0, 0, 0],
        [0, 0, 0, 0, 0, _min_prime(x[5],x[10]), 0, 0, 0, 0, _min_prime(x[10],x[5]), 0, 0, 0],
        [0, 0, 0, 0, 0, 0, _min_prime(x[6],x[11]), 0, 0, 0, 0, _min_prime(x[11],x[6]), 0, 0],
        [0, 0, 0, 0, 0, 0, 0, _min_prime(x[7],x[12]), 0, 0, 0, 0, _min_prime(x[12],x[7]), 0],
        [0, 0, 0, 0, 0, 0, 0, 0, _min_prime(x[8],x[13]), 0, 0, 0, 0, _min_prime(x[13],x[8])]
    ]),
    lambda x: [x[4:]>=0],
    lambda x: p4_proj(x),
    np.array([1, 4, -2, 1, 3, 3, 1, 4, 1, 0, 1, 3, 1, 3])
)

def p5(n: int):
    return Problem(
        lambda x: (x+1)**2 + x - (2/n) * np.sum(x) - 1,
        lambda x: np.diag(2 * x + 3) - 2 / n,
        lambda x: [x<=10, x>=-10],
        lambda x: np.clip(x, -10, 10),
        np.ones(n)
    )

def mon1(n: int):
    return Problem(
        lambda x: 2e-5 * (x - 1) + 4 * (np.linalg.norm(x, axis=0)**2 - 0.25) * x,
        # lambda x: 8 * np.outer(x, x) + 2e-5 + 4*(np.linalg.norm(x)**2 - 0.25) * np.diag(np.ones(n)),
        lambda x: 8 * np.outer(x, x) + np.diag(np.full((n,), 2e-5 + 4 * (np.linalg.norm(x)**2 - 0.25))),
        lambda x: [x>=0],
        lambda x: np.maximum(x, 0),
        np.ones(n)
    )

def mon2_derivative(x, n):
    shifts = np.empty_like(x)
    shifts[0] = (x[0] + x[1]) / 2
    shifts[1:-1] = (x[:-2] + x[1:-1] + x[2:]) / np.arange(2, n)
    shifts[-1] = (x[-2] + x[-1]) / n
    diags = np.empty_like(x)
    diags = np.exp(np.cos(shifts)) * np.sin(shifts) / np.arange(1, n+1)
    diags[0] /= 2
    G = ssp.diags([diags[:-1], 1+diags, diags[1:]], [-1, 0, 1])
    return G.toarray() # don't use sparse matrices as cvxpy solver falls apart terribly https://github.com/cvxpy/cvxpy/issues/1159

def mon2(n: int):
    return Problem(
        lambda x: np.concatenate((
            x[0, np.newaxis] - np.exp(np.cos((x[0, np.newaxis] + x[1, np.newaxis])/2)),
            x[1:-1] - np.exp(np.cos((x[:-2] + x[1:-1] + x[2:]) / np.arange(2, n))),
            x[-1, np.newaxis] - np.exp(np.cos((x[-1, np.newaxis] + x[-2, np.newaxis]))/n)
        )),
        lambda x: mon2_derivative(x, n),
        lambda x: [x >= 0],
        lambda x: np.maximum(x, 0),
        np.ones(n)
    )

def mon3(n: int):
    return Problem(
        lambda x: np.log(x + 1) - x / n,
        lambda x: np.diag(1 / (x + 1) - 1/n),
        lambda x: [x >= 0],
        lambda x: np.maximum(x, 0),
        np.ones(n)
    )

def mon4(n: int):
    return Problem(
        lambda x: 2 * x - np.sin(np.abs(x)),
        lambda x: np.diag(2 + np.piecewise(x, [x>=0], [lambda x: np.cos(x), lambda x: -np.cos(x)])),
        lambda x: [x >= 0],
        lambda x: np.maximum(x, 0),
        np.ones(n)
    )

def mon5(n: int):
    return Problem(
        lambda x: np.exp(x) - 1,
        lambda x: np.diag(np.exp(x)),
        lambda x: [x >= 0],
        lambda x: np.maximum(x, 0),
        np.ones(n)
    )

def mon6_proj(x):
    y = np.maximum(x, 0)
    n = x.size
    # Project onto the simplex
    # Taken from https://gist.github.com/mblondel/6f3b7aaad90606b98f71
    if np.sum(y) > n:
        z = np.sort(y)[::-1]
        s = np.cumsum(z) - n
        ind = np.arange(1, n+1)
        cond = z - s / ind > 0
        k = ind[cond][-1]
        tau = s[cond][-1] / k
        y = np.maximum(y - tau, 0)
    return y

def mon6(n: int):
    return Problem(
        lambda x: x - np.sin(np.abs(x - 1)),
        lambda x: np.diag(1 - np.piecewise(x, [x>=1], [lambda x: np.cos(x-1), lambda x: -np.cos(1-x)])),
        # lambda x: [x >= 0, np.sum(x) <= n],
        lambda x: [x >= 0, x @ np.ones(n) <= n],
        lambda x: mon6_proj(x),
        np.ones(n)
    )
