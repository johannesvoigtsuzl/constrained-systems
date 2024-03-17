import numpy as np
from matplotlib import pyplot as plt
from time import monotonic
import examples as ex
import cg as CG
import newton as nw
import os.path
from itertools import product

def test_solver_residuals(solver, kmax = 1000, res_tol=1e-10, step_tol=1e-15):
    t = monotonic()
    residuals = []
    res = step = np.inf
    k = 0
    for k, (res, step) in enumerate(solver.run()):
        residuals.append(res)
        print(f"k = {k:3d}: {res:e}")
        if k >= kmax or res < res_tol or step < step_tol:
            break

    t = monotonic() - t
    print(f"Took {t:.5f} seconds")

    return np.array(residuals)

seed = 4242

def cached_residuals(path, solver, solver_opts):
    if os.path.isfile(path):
        with open(path, "rb") as f:
            residuals = np.load(f)
            return residuals
    else:
        residuals = test_solver_residuals(solver, **solver_opts)
        with open(path, "wb") as f:
            np.save(f, residuals)
            return residuals

def plot_cg_residuals():
    save_fig = False
    cache = False

    solvers = {
        "NDK1": CG.cg_method_ndk1,
        "MDKM": CG.cg_method_mdkm,
        "Gao-He": CG.cg_method_gao_he,
        "GCD": CG.cg_method_cgd,
        "MHZ1": CG.cg_method_mhz1,
        "MFRM": CG.cg_method_mfrm,
        "HSG": CG.sg_hsg,
    }

    probs = {
        "mon1": ex.mon1,
        "mon2": ex.mon2,
        "mon3": ex.mon3,
        "mon4": ex.mon4,
        "mon5": ex.mon5,
        "mon6": ex.mon6,
    }

    n = 50_000

    solver_opts = {
        "kmax": 1000,
        "res_tol": 1e-10,
        "step_tol": 1e-16,
    }

    rng = np.random.default_rng(seed)

    x0 = rng.uniform(0, 5, (n,))
    x0_mon5 = rng.uniform(0, 1, (n,))

    for prob_name, prob in probs.items():
        prob = prob(n)
        prob.z0 = x0_mon5 if prob_name == "mon5" else x0
        fig, ax = plt.subplots(1, 3, figsize = (12, 4))
        for solver_name, solver in solvers.items():
            res = None
            if cache:
                path = f"data/{solver_name}/seed{seed}/res-{prob_name}.npy"
                res = cached_residuals(path, solver(prob), solver_opts)
            else:
                res = test_solver_residuals(solver(prob), **solver_opts)

            ax[0].plot(res, label=solver_name, linewidth=1)
            ax[0].legend(loc='upper right')
            ax[1].semilogy(res, linewidth=1)
            ax[2].plot(res[1:] / res[:-1], linewidth=1)

        if save_fig:
            fig.savefig(f"images/residuals-cg-{prob_name}.png")
        plt.show()

def main():
    plt.tight_layout()
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.bbox'] = "tight"
    plt.close("all")

    plot_cg_residuals()

    # gen_latex_tables()

if __name__ == '__main__':
    main()
