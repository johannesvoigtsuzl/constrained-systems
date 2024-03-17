import numpy as np
from matplotlib import pyplot as plt
from time import monotonic
import examples as ex
import cg as CG
import newton as nw
import os.path

from cg_residuals import test_solver_residuals

def test_solver(solver, kmax = 100, res_tol=1e-15, step_tol=1e-16):
    suc = False
    t = monotonic()
    residuals = []
    res = step = np.inf
    k = 0
    for k, (res, step) in enumerate(solver.run()):
        residuals.append(res)
        print(f"k = {k:3d}: {res:e}", end="\r")
        if k >= kmax or res < res_tol or step < step_tol:
            print(f"k = {k:3d}: {res:e}")
            break

    t = monotonic() - t
    print(f"Took {t:.5f} seconds")
    evals = solver.evals
    if residuals[-1] < res_tol or step < step_tol:
        suc = True

    return { "success": suc, "k": k, "evals": evals, "time": t, "res": res }, np.array(residuals)

def average_jacobian_diag(f, a, b):
    return f(b) - f(a) / (b - a)

def test_assumption_smlp(prob, avg_jac_func):
    res_tol = 1e-15
    step_tol = 1e-12

    smlp = nw.smlp_newton(prob)

    diff_z = []
    diff_mat = []

    for k, (res, step) in enumerate(smlp.run()):
        if k > 0:
            diff_z.append(np.linalg.norm(smlp.z - smlp.z_old))
            diff_mat.append(np.linalg.norm(avg_jac_func(prob.f, smlp.z_old, smlp.z) - smlp.M))
        print(f"k = {k:2d}: {res:e}")
        if k >= 99 or res < res_tol or step < step_tol:
            break

    fig = plt.figure()
    ax = fig.add_subplot()
    q = np.log(np.array(diff_mat)) / np.log(np.array(diff_z))
    ax.plot(q, linewidth = 1)
    ax.hlines(0, 0, len(diff_z)-1, linewidth = 1, color="red")
    return fig

def plot_lp_residuals(prob_name, prob, solver_opts, save_fig=False):
    fig, ax = plt.subplots(2, 2)
    _, res_lp   = test_solver(nw.lp_newton(prob), **solver_opts)
    _, res_smlp = test_solver(nw.smlp_newton(prob), **solver_opts)
    print()
    ax[0, 0].plot(res_lp, label="LP", linewidth=1)
    ax[0, 0].plot(res_smlp, label="SMLP", linewidth=1)
    ax[0, 0].legend(loc='upper right')
    ax[0, 1].semilogy(res_lp, linewidth=1)
    ax[0, 1].semilogy(res_smlp, linewidth=1)
    ax[1, 0].plot(res_lp[1:] / res_lp[:-1]**2, linewidth=1)
    ax[1, 1].plot(res_smlp[1:] / res_smlp[:-1], linewidth=1, color = "orange")

    if save_fig:
        fig.savefig(f"images/residuals-lp-{prob_name}.png")
    plt.show()

def gen_latex_table(solver_opts):
    n = 50

    probs = {
        "mon1": ex.mon1(n),
        "mon3": ex.mon3(n),
        "mon4": ex.mon4(n),
        "mon5": ex.mon5(n),
        "mon6": ex.mon6(n),
        "p1":   ex.p1,
        "p2":   ex.p2,
        "p3":   ex.p3,
        "p4":   ex.p4,
        "p5":   ex.p5(n),
    }

    text = "\\begin{tabular}{l" + "|rcc"*2 + "}\n"
    text += "\t\\multicolumn{1}{c}{} & \\multicolumn{3}{|c}{LP-Newton}& \\multicolumn{3}{|c}{SMLP-Newton} \\\\\n\t\\hline\n"
    text += "\tProblem" + " & Iters & Time & $\\|F(x^*)\\|$" * 2 + " \\\\\n\\hline\n"

    for prob_name, prob in probs.items():
        text += f"\t\\cref{{ex:num:{prob_name}}}"
        for s in [nw.lp_newton, nw.smlp_newton]:
            result, _ = test_solver(s(prob), **solver_opts)

            res = f"{result['res']:.2e}"
            man, e = map(str.strip, res.split('e'))
            if result['res'] < 1e-1 and int(e) == 0:
                # This happens exactly when residual == 0
                res = "0"
            else:
                res = f"{man}\\times 10^{{{int(e):3d}}}"

            text += f" & {result['k']:3d} & {result['time']:.3f} & ${res}$"
        text += "\\\\\n"
    text += "\\end{tabular}\n"

    os.makedirs(f"thesis/tables", exist_ok=True)
    with open(f"thesis/tables/lp.tex", "w") as tex_file:
        print(text, file=tex_file)


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

def main():
    plt.tight_layout()
    plt.rcParams['figure.dpi'] = 150
    plt.close("all")

    solver_opts = {
        "kmax": 100,
        "res_tol": 1e-10,
        "step_tol": 1e-16,
    }

    # gen_latex_table(solver_opts)

    probs = {
        "p1": ex.p1,
        "p2": ex.p2,
        "p3": ex.p3,
        "p4": ex.p4,
        "p5": ex.p5(50),
        "mon1": ex.mon1(50),
        # skip "mon2"
        "mon3": ex.mon3(50),
        "mon4": ex.mon4(50),
        "mon5": ex.mon5(50),
        "mon6": ex.mon6(50),
    }

    for prob_name, prob in probs.items():
        plot_lp_residuals(prob_name, prob, solver_opts, save_fig=True)


if __name__ == '__main__':
    main()
