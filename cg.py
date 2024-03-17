import numpy as np
from typing import Callable, List
import examples as ex

class cg_base:
    f: Callable[[np.ndarray], np.ndarray]
    omega: Callable[[np.ndarray], List[np.ndarray]]
    x: np.ndarray
    proj: Callable[[np.ndarray], np.ndarray]
    d: np.ndarray
    f_x: np.ndarray
    tol: float
    evals: int = 0
    stop: bool = False
    # Usually 1, as not all methods have this parameter
    kappa: float = 1
    step_size: float = float("inf")

    def __init__(self, prob: ex.Problem, tol=1e-16):
        self.x = prob.proj(prob.z0)
        self.f = prob.f
        self.omega = prob.omega
        self.proj = prob.proj
        self.f_x = self.f(self.x)
        self.d = -self.f_x
        self.evals += 1
        self.tol = tol

    def step(self):
        raise NotImplementedError("This intended to only be a base class")

    def run(self):
        yield np.linalg.norm(self.f_x, ord=np.inf), float("inf")
        while not self.stop:
            self.step()
            yield np.linalg.norm(self.f_x, ord=np.inf), self.step_size

    def get_iterate(self):
        return self.x

class cg_method_ndk1(cg_base):
    sigma = 0.5
    rho = 0.5

    r = 0.1
    G = 1
    q = 1.6
    phi = 1.9

    def __init__(self, prob: ex.Problem, **kwargs):
        cg_base.__init__(self, prob, **kwargs)

    def step(self):
        dd = np.linalg.norm(self.d)**2

        alpha = self.kappa
        z = self.x + alpha * self.d
        f_z = self.f(z)
        self.evals += 1
        while not -np.inner(self.d, f_z) >= self.sigma * alpha * dd:
            alpha *= self.rho
            z = self.x + alpha * self.d
            f_z = self.f(z)
            self.evals += 1
            if np.linalg.norm(f_z) < self.tol and all(map(np.all, self.omega(z))):
                print("Early exit!")
                self.stop = True
                self.x = z
                self.f_x = f_z
                return

        x_new = self.x - self.phi * np.inner(f_z, self.x - z) / np.linalg.norm(f_z)**2 * f_z
        x_new = self.proj(x_new)
        self.step_size = np.linalg.norm(x_new - self.x)

        f_x_new = self.f(x_new)
        self.evals += 1

        # Calculate tau
        s = z - self.x
        y = f_z - self.f_x
        rho_k = max(-np.inner(s, y) / np.linalg.norm(s)**2, 0.0)
        y += rho_k * s + self.G * np.linalg.norm(self.f_x)**self.r * s
        sy = np.inner(s, y)
        dy = np.inner(self.d, y)
        ss = np.linalg.norm(s)**2
        yy = np.linalg.norm(y)**2

        tau_tilde = 3 * sy / ss - yy / sy
        tau = max(tau_tilde, self.q * yy / sy)

        # Calculate beta
        fs = np.inner(f_x_new, s)
        beta = np.inner(f_x_new, y) / dy - (tau + yy / sy - sy / ss) * fs / dy

        self.x = x_new
        self.f_x = f_x_new
        self.d = -self.f_x + beta * self.d + fs / sy * y

# A modified Dai–Kou-type method with applications to signal reconstruction and blurred image restoration - Waziri, Ahmed, Halilu 2022
class cg_method_mdkm(cg_base):
    sigma = 1e-2
    rho = 0.8

    psi = 1e-3
    r = 0.1
    E = 1.1

    def __init__(self, prob: ex.Problem, **kwargs):
        cg_base.__init__(self, prob, **kwargs)
        self.kappa = 0.9

    def step(self):
        dd = np.linalg.norm(self.d)**2

        alpha = self.kappa
        z = self.x + alpha * self.d
        f_z = self.f(z)
        self.evals += 1
        while not -np.inner(self.d, f_z) >= self.sigma * alpha * dd:
            alpha *= self.rho
            z = self.x + alpha * self.d
            f_z = self.f(z)
            self.evals += 1
            if np.linalg.norm(f_z) < self.tol and all(map(np.all, self.omega(z))):
                print("Early exit!")
                self.stop = True
                self.x = z
                self.f_x = f_z
                return

        x_new = self.x - np.inner(f_z, self.x - z) / np.linalg.norm(f_z)**2 * f_z
        x_new = self.proj(x_new)
        self.step_size = np.linalg.norm(x_new - self.x)

        f_x_new = self.f(x_new)
        self.evals += 1

        s = z - self.x
        ss = np.linalg.norm(s)**2
        y = f_z - self.f_x
        f_norm = np.linalg.norm(self.f_x)
        f_new_norm = np.linalg.norm(f_x_new)
        lamb = max(-np.inner(s, y) / ss, 0)

        y += (lamb + self.E * f_norm**self.r) * s
        sy = np.inner(s, y)
        dy = np.inner(self.d, y)
        yy = np.linalg.norm(y)**2

        tau = 1 + sy / ss - yy / sy

        # Calculate beta
        fy = np.inner(f_x_new, y)
        fs = np.inner(f_x_new, s)
        fd = np.inner(f_x_new, self.d)
        beta = fy / dy - (tau + yy / sy - sy / ss) * fs / dy
        beta = max(beta, self.psi * f_new_norm / np.sqrt(dd))

        self.x = x_new
        self.f_x = f_x_new
        self.d = -self.f_x + beta * (self.d - fd / f_new_norm**2 * self.f_x)

# Xiao, Zhu 2013
class cg_method_cgd(cg_base):
    sigma = 0.5
    rho = 0.5

    r = 0.1
    G = 1
    q = 1.6
    phi = 1.9

    def __init__(self, prob: ex.Problem, **kwargs):
        cg_base.__init__(self, prob, **kwargs)

    def step(self):
        dd = np.linalg.norm(self.d)**2

        alpha = self.kappa
        z = self.x + alpha * self.d
        f_z = self.f(z)
        self.evals += 1
        while not -np.inner(self.d, f_z) >= self.sigma * alpha * dd:
            alpha *= self.rho
            z = self.x + alpha * self.d
            f_z = self.f(z)
            self.evals += 1
            if np.linalg.norm(f_z) < self.tol and all(map(np.all, self.omega(z))):
                print("Early exit!")
                self.stop = True
                self.x = z
                self.f_x = f_z
                return

        x_new = self.x - np.inner(f_z, self.x - z) / np.linalg.norm(f_z)**2 * f_z
        x_new = self.proj(x_new)
        self.step_size = np.linalg.norm(x_new - self.x)

        f_x_new = self.f(x_new)
        self.evals += 1

        f_x_norm = np.linalg.norm(self.f_x)
        y = f_x_new - self.f_x
        dy = np.inner(self.d, y)
        lambda_k = 1 + max(0, -dy / alpha*dd) / f_x_norm
        y += lambda_k * f_x_norm * self.d

        dy = np.inner(self.d, y)
        yy = np.linalg.norm(y)**2


        # Calculate beta
        fac = 2 * yy / dy
        beta = np.inner(f_x_new, y - fac * self.d) / dy

        self.x = x_new
        self.f_x = f_x_new

        self.d = -self.f_x + beta * self.d

# An efficient three-term conjugate gradient method for nonlinear monotone equations with convex constraints - Gao, He 2018
class cg_method_gao_he(cg_base):
    sigma = 1e-4
    rho = 0.4

    gamma = 1

    def __init__(self, prob: ex.Problem, **kwargs):
        cg_base.__init__(self, prob, **kwargs)

    def step(self):
        dd = np.linalg.norm(self.d)**2

        alpha = self.kappa
        z = self.x + alpha * self.d
        f_z = self.f(z)
        self.evals += 1
        while not -np.inner(self.d, f_z) >= self.sigma * alpha * np.linalg.norm(f_z) * dd:
            alpha *= self.rho
            z = self.x + alpha * self.d
            f_z = self.f(z)
            self.evals += 1

        x_new = self.x - self.gamma * np.inner(f_z, self.x - z) / np.linalg.norm(f_z)**2 * f_z
        x_new = self.proj(x_new)
        self.step_size = np.linalg.norm(x_new - self.x)

        # Note: Only Gao-He mentions the case F(z_k) < eps && z_k \notin \Omega
        if np.linalg.norm(f_z) < self.tol:
            if all(map(np.all, self.omega(z))):
                print()
                print("Early exit!")
                self.stop = True
                self.x = z
                self.f_x = f_z
                return
            else:
                print()
                print("Project z!")
                x_new = self.proj(z)

        f_x_new = self.f(x_new)
        self.evals += 1

        fd = np.inner(self.f_x, self.d)
        f_f_new = np.inner(self.f_x, f_x_new)
        d_f_new = np.inner(self.d, f_x_new)

        beta = -f_f_new / fd
        theta = d_f_new / fd

        self.d = -f_x_new + beta * self.d + theta * self.f_x
        self.x = x_new
        self.f_x = f_x_new

# A modified Hager-Zhang conjugate gradient method with optimal choices for solving monotone nonlinear equations - Sabi'u, Shah, Waziri 2021
class cg_method_mhz1(cg_base):
    sigma = 1e-4
    delta = 0.3

    rho = 0.5 # Not given in paper

    def __init__(self, prob: ex.Problem, **kwargs):
        cg_base.__init__(self, prob, **kwargs)

    def step(self):
        dd = np.linalg.norm(self.d)**2

        alpha = self.kappa
        z = self.x + alpha * self.d
        f_z = self.f(z)
        self.evals += 1
        tries = 0
        while not -np.inner(self.d, f_z) >= self.sigma * alpha * dd:
            alpha *= self.delta
            z = self.x + alpha * self.d
            f_z = self.f(z)
            self.evals += 1
            if np.linalg.norm(f_z) < self.tol and all(map(np.all, self.omega(z))):
                print("Early exit!")
                self.stop = True
                self.x = z
                self.f_x = f_z
                return
            tries += 1
            if tries > 300:
                print("Timeout in Backtracking")
                self.stop = True
                return

        x_new = self.x - np.inner(f_z, self.x - z) / np.linalg.norm(f_z)**2 * f_z
        x_new = self.proj(x_new)
        self.step_size = np.linalg.norm(x_new - self.x)

        f_x_new = self.f(x_new)
        self.evals += 1

        s = z - self.x
        ss = np.linalg.norm(s)**2
        y = f_z - self.f_x
        theta = 3 * (np.linalg.norm(self.f_x)**2 - np.linalg.norm(f_z)**2) + 3 * np.inner(self.f_x + f_z, s)
        y += self.rho + max(theta, 0) / ss * s
        sy = np.inner(s, y)
        dy = np.inner(self.d, y)
        yy = np.linalg.norm(y)**2
        fy = np.inner(f_x_new, y)
        fd = np.inner(f_x_new, self.d)

        beta_hs = fy / dy
        q = sy**2 / ss / yy
        theta = q + np.sqrt(q)

        beta = beta_hs - theta * (yy * fd) / dy**2

        self.x = x_new
        self.f_x = f_x_new
        self.d = -self.f_x + beta * self.d

# A Modified Fletcher–Reeves Conjugate Gradient Method for Monotone Nonlinear Equations with Some Applications - Abukabar, Kumam, Mohammad, Awwal, Sitthithakerngkiet 2019
class cg_method_mfrm(cg_base):
    sigma = 1e-4
    rho = 0.9

    mu = 1e-2

    def __init__(self, prob: ex.Problem, **kwargs):
        cg_base.__init__(self, prob, **kwargs)

    def step(self):
        dd = np.linalg.norm(self.d)**2
        f_norm = np.linalg.norm(self.f_x)

        alpha = self.kappa
        z = self.x + alpha * self.d
        # TODO: check if ||f_z|| < eps
        f_z = self.f(z)
        self.evals += 1
        while not -np.inner(self.d, f_z) >= self.sigma * alpha * np.linalg.norm(f_z) * dd:
            alpha *= self.rho
            z = self.x + alpha * self.d
            f_z = self.f(z)
            self.evals += 1
            if np.linalg.norm(f_z) < self.tol and all(map(np.all, self.omega(z))):
                print("Early exit!")
                self.stop = True
                self.x = z
                self.f_x = f_z
                return

        if np.linalg.norm(f_z) < self.tol:
            if all(map(np.all, self.omega(z))):
                print()
                print("Early exit!")
                self.stop = True
                self.x = z
                self.f_x = f_z
                return
            else:
                print()
                print("Project z!")
                x_new = self.proj(z)

        x_new = self.x - np.inner(f_z, self.x - z) / np.linalg.norm(f_z)**2 * f_z
        x_new = self.proj(x_new)
        self.step_size = np.linalg.norm(x_new - self.x)

        f_x_new = self.f(x_new)
        self.evals += 1

        s = z - self.x
        s_norm = np.linalg.norm(s)
        f_new_norm = np.linalg.norm(f_x_new)

        factor = max(self.mu * s_norm * f_new_norm, f_norm**2)

        self.x = x_new
        self.f_x = f_x_new
        self.d = -self.f_x + f_new_norm**2/factor * s - np.inner(f_x_new, s)/factor * f_x_new

# A new hybrid spectral gradient projection method for monotone system of nonlinear equations with convex constraints - Awwal, Kuman et al 2018
class sg_hsg(cg_base):
    sigma = 1e-3
    rho = 0.9
    r = sigma

    def __init__(self, prob: ex.Problem, **kwargs):
        cg_base.__init__(self, prob, **kwargs)

    def step(self):
        dd = np.linalg.norm(self.d)**2
        f_norm = np.linalg.norm(self.f_x)

        alpha = self.kappa
        z = self.x + alpha * self.d
        f_z = self.f(z)
        self.evals += 1
        while not -np.inner(self.d, f_z) >= self.sigma * alpha * dd:
            alpha *= self.rho
            z = self.x + alpha * self.d
            f_z = self.f(z)
            self.evals += 1
            if np.linalg.norm(f_z) < self.tol and all(map(np.all, self.omega(z))):
                print("Early exit!")
                self.stop = True
                self.x = z
                self.f_x = f_z
                return

        x_new = self.x - np.inner(f_z, self.x - z) / np.linalg.norm(f_z)**2 * f_z
        x_new = self.proj(x_new)

        f_x_new = self.f(x_new)
        self.evals += 1

        s = x_new - self.x
        self.step_size = np.linalg.norm(s)
        y = f_x_new - self.f_x
        v = y + self.r * s
        if self.step_size < 1e-17:
            print()
            print("Division by 0 in HSG, disqualified")
            self.stop = True
            return

        lamb = self.step_size**2 / np.inner(v, s)
        gamma = self.step_size / np.linalg.norm(v)
        f_new_norm = np.linalg.norm(f_x_new)
        fd = np.inner(f_x_new, self.d)
        theta = 1 - fd**2 / (f_new_norm**2 * dd)
        tau = (1 - theta) * lamb + theta * gamma

        self.x = x_new
        self.f_x = f_x_new
        self.d = -tau * self.f_x
