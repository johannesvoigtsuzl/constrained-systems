import numpy as np
from matplotlib import pyplot as plt
from time import monotonic
import examples as ex
import cg as CG
import newton as nw
import os.path
from itertools import product

def test_solver(solver, kmax = 1000, res_tol=1e-10, step_tol=1e-15):
    suc = False
    t = monotonic()
    res = step = np.inf
    k = 0
    for k, (res, step) in enumerate(solver.run()):
        print(f"k = {k:3d}: {res:e}", end="\r")
        if k >= kmax or res < res_tol or step < step_tol:
            print(f"k = {k:3d}: {res:e}")
            break

    t = monotonic() - t
    print(f"Took {t:.5f} seconds")
    evals = solver.evals
    if res < res_tol or step < step_tol:
        suc = True

    solution = solver.get_iterate()
    return { "success": suc, "k": k, "evals": evals, "time": t , "res": res, "step": step }, solution

def make_performance_profile(measurements: np.ndarray, resolution=100, t_max=5):
    # fist axis: Solver, second axis: Problem
    max = np.min(measurements, axis=0)
    n_p = measurements.shape[1]

    # If all solvers fail at a problem, we encounter inf/inf
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.nan_to_num(measurements / max, nan=np.inf, copy=False)
    t = np.linspace(1, t_max, resolution)
    p = np.sum(np.repeat(r[..., np.newaxis], resolution, axis = -1) <= t, axis=1) / n_p
    return t, p

seed = 4242

def cached_result(path, solver, solver_opts, verbose=True):
    result = {}
    if os.path.isfile(path):
        with open(path, "rb") as f:
            file = np.load(f, mmap_mode="r")
            result = { key: file[key] for key in ["success", "k", "evals", "time", "res", "step"] }
            k, res, t = result["k"], file["res"], result["time"]
            if verbose:
                print("CACHED")
                print(f"k = {k:3d}: {res:e}")
                print(f"Took {t:.5f} seconds")
            file.close()
    else:
        result, solution = test_solver(solver, **solver_opts)
        with open(path, "wb") as f:
            np.savez_compressed(f, **result, solution=solution)
    return result

def benchmark_cg(solvers, probs_iter, field, solver_opts, cache=False):
    results = {}
    for name in solvers.keys():
        results[name] = []
        os.makedirs(f"data/{name}/seed{seed}", exist_ok=True)

    for file_basename, prob in probs_iter:
        for name, solver in solvers.items():
            path = f"data/{name}/{file_basename}.npz"
            print(name)
            result = (cached_result(path, solver(prob), solver_opts)
                      if cache
                      else test_solver(solver(prob), **solver_opts)[0])
            results[name].append(result[field] if result["success"] else np.inf)
        print()
    return results

# We let a generator initiate our problems iteratively to not keep
# all of them in memory at the same time
def generate_cg_problems(probs, ns, runs):
    for n in ns:
        for prob_name, prob in [(name, prob(n)) for (name, prob) in probs.items()]:
            # Reset generator for each problem to ensure the initial values remain consistent
            rng = np.random.default_rng(seed)
            for i in range(runs):
                print(f"n = {n}; Problem {prob_name}; x{i}")
                if prob_name == "mon5":
                    prob.z0 = rng.uniform(0, 1, prob.z0.shape)
                else:
                    prob.z0 = rng.uniform(0, 5, prob.z0.shape)
                file_basename = f"seed{seed}/{prob_name}-n{n}-x{i}"
                yield file_basename, prob

# To make my life much easier
def gen_single_latex_table(solvers, ns, n_xs, prob_name, prob, solver_opts):
    text = ""

    names = ""
    names_file = ""
    text += "\t\\multicolumn{2}{c}{}"
    for k, name in enumerate(solvers.keys()):
        os.makedirs(f"data/{name}/seed{seed}", exist_ok=True)
        if k+2 < len(solvers):
            names += name + ", "
            names_file += name + "-"
        elif k+2 == len(solvers):
            names += name + " and "
            names_file += name + "-"
        else:
            names += name
            names_file += name
        text += f" & \\multicolumn{{4}}{{c}}{{{name}}}"

    text += " \\\\\n\t\\hline"

    text += "\t$n$ & Start" + " & Iters & Evals & Time & $\\|F(x^*)\\|$"*len(solvers) + " \\\\\n"
    for n, i in product(ns, range(n_xs)):
        if i == 0:
            text += "\t\\hline\n"
        file_basename = f"seed{seed}/{prob_name}-n{n}-x{i}"
        row = f"\t{n:6d} & $x_{i}$"
        for name, solver in solvers.items():
            path = f"data/{name}/{file_basename}.npz"
            # We did not properly generate x0
            if not os.path.isfile(path):
                return
            result = cached_result(path, solver(prob(n)), solver_opts, verbose=False)
            res = f"{result['res']:.2e}"
            man, e = map(str.strip, res.split('e'))
            row += f" & {result['k']:4d} & {result['evals']:4d} & {result['time']:.3f} & ${man}\\times 10^{{{int(e):3d}}}$"
        text += row + " \\\\\n"

    os.makedirs(f"thesis/tables", exist_ok=True)
    with open(f"thesis/tables/{prob_name}-{names_file}.tex", "w") as tex_file:
        print(text, file=tex_file)
        # print(text)

def gen_latex_tables():
    ns = [1000, 5000, 10_000, 50_000, 100_000]

    solver_opts = {
        "kmax": 1000,
        "res_tol": 1e-15,
        "step_tol": 1e-16
    }

    solvers1 = {
        "NDK1": CG.cg_method_ndk1,
        "MDKM": CG.cg_method_mdkm,
    }
    solvers2 = {
        "Gao-He": CG.cg_method_gao_he,
        "GCD": CG.cg_method_cgd,
    }
    solvers3 = {
        "MHZ1": CG.cg_method_mhz1,
        "MFRM": CG.cg_method_mfrm,
    }
    solvers4 = {
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

    for prob_name, prob in probs.items():
        gen_single_latex_table(solvers1, ns, 3, prob_name, prob, solver_opts)
        gen_single_latex_table(solvers2, ns, 3, prob_name, prob, solver_opts)
        gen_single_latex_table(solvers3, ns, 3, prob_name, prob, solver_opts)
        gen_single_latex_table(solvers4, ns, 3, prob_name, prob, solver_opts)

def main():
    plt.tight_layout()
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.bbox'] = "tight"
    plt.close("all")

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

    ns = [1000, 5000, 10_000, 50_000, 100_000]
    runs = 3

    cache = False
    save_fig = False

    solver_opts = {
        "kmax": 1000,
        "res_tol": 1e-8,
        "step_tol": 1e-14
    }

    for field in ["evals", "k", "time"]:
        results = benchmark_cg(
            solvers,
            generate_cg_problems(probs, ns, runs),
            field,
            solver_opts,
            cache=cache
        )

        fig = plt.figure()
        fig.tight_layout()
        ax = fig.add_subplot()

        t, profile = make_performance_profile(np.array(list(results.values())), resolution=1000, t_max=10)
        for i, name in enumerate(results.keys()):
            plt.step(t, profile[i], where="post", label=name, linewidth=1)
        # Python, why is it plt.xlim, but then ax.set_xlim ???
        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right')
        if save_fig:
            fig.savefig(f"images/perf-{field}.png")
        plt.show()

    if cache == True:
        gen_latex_tables()


if __name__ == '__main__':
    main()
