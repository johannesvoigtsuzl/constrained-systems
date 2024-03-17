#!/usr/bin/env python

import cvxpy as cp
import numpy as np
from typing import Callable
import examples as ex

class lp_subproblem:
    def __init__(self, n, m, omega, solver='SCIPY', verbose=False):
        self.verbose = verbose
        self.solver = solver

        self.s = cp.Parameter(n)
        self.v = cp.Parameter(m)
        self.M = cp.Parameter((m, n))
        self.a1 = cp.Parameter(nonneg=True)
        self.a2 = cp.Parameter(nonneg=True)
        self.gamma = cp.Variable(nonneg=True)
        self.z = cp.Variable(n)

        self.prob = cp.Problem(cp.Minimize(self.gamma), [
            self.M @ self.z <= self.a1 * self.gamma + self.v,
            -self.M @ self.z <= self.a1 * self.gamma - self.v,
            self.s - self.gamma * self.a2<= self.z,
            self.z <= self.s + self.gamma * self.a2,
            *omega(self.z)
        ])

    def step(self, s, y, M, a1, a2=1.):
        self.s.value = s
        self.v.value = M @ s - y
        self.M.value = M
        self.a1.value = a1
        self.a2.value = a2

        self.prob.solve(solver = self.solver, verbose=self.verbose, ignore_dpp=True, scipy_options={'presolve':False,'disp':False,'method':'highs-ipm'})
        return self.z.value, self.gamma.value

class lp_newton:
    f: Callable[[np.ndarray], np.ndarray]
    G: Callable[[np.ndarray], np.ndarray]
    z: np.ndarray
    p: np.ndarray
    y: np.ndarray
    lp: lp_subproblem
    # Only exists for compatibility's sake with CG solver
    evals: int = 0

    def __init__(self, prob: ex.Problem, solver='SCIPY', verbose=False):
        self.f = prob.f
        self.G = prob.G
        self.z = prob.proj(prob.z0)

        self.y = self.f(self.z)
        self.evals +=1
        n = self.z.size
        m = self.y.size

        self.lp = lp_subproblem(n, m, prob.omega, solver=solver, verbose=verbose)

    def step(self):
        z_old = self.z
        M = self.G(self.z)
        a = float(np.linalg.norm(self.y, ord=np.inf))
        self.z, _ = self.lp.step(self.z, self.y, M, a**2, a)
        self.y = self.f(self.z)
        self.evals += 1
        self.p = self.z - z_old

    def run(self):
        yield np.linalg.norm(self.y, ord=np.inf), float("inf")
        while True:
            self.step()
            yield np.linalg.norm(self.y, ord=np.inf), np.linalg.norm(self.p)

    def get_iterate(self):
        return self.z

class smlp_newton(lp_newton):
    def __init__(self, prob:ex.Problem):
        lp_newton.__init__(self, prob)
        self.M = self.G(prob.z0)
        self.eta0 = np.linalg.norm(self.y, ord=np.inf) / self.z.size**2
        self.eta = self.eta0

    def step(self):
        self.z_old = self.z
        y_old = self.y

        self.z, _ = self.lp.step(self.z, self.y, self.M, self.eta)
        self.y = self.f(self.z)
        self.evals += 1

        self.eta = np.minimum(self.eta0, 100*np.maximum(np.linalg.norm(y_old, ord=np.inf), np.linalg.norm(self.y, ord=np.inf)))

        self.p = self.z - self.z_old
        step_length = np.linalg.norm(self.p)
        self.M += np.outer(self.y - y_old - self.M @ self.p, self.p) / step_length**2
