"""Algorithm 1 verification — trimmed for fast run."""
import numpy as np, time
from scipy.optimize import differential_evolution
from scipy.integrate import quad

PI = np.pi
LW = {1: 50.0, 2: 30.0, 3: 10.0, 4: 3.0}


def gsig(theta, T=1.0):
    t1, b1, b2 = theta
    t2 = T - t1
    return {
        (1,): T,
        (2,): b1*t1 + b2*t2,
        (1,2): 0.5*b1*t1**2 + 0.5*b2*t2**2 + b2*t1*t2,
        (1,1,2): (1/6)*b1*t1**3 + (1/6)*b2*t2**3 + 0.5*b2*t1*t2**2 + 0.5*b2*t1**2*t2,
        (1,2,2): ((1/6)*b1**2*t1**3 + (1/6)*b2**2*t2**3
                  + 0.5*b2**2*t1*t2**2 + 0.5*b1*b2*t1**2*t2),
    }


def J(th, target):
    g = gsig(th)
    return sum(LW[len(w)] * (g[w]-tv)**2 for w, tv in target.items())


def run_de(target, n_seeds=15, popsize=15, maxiter=100, bounds=None):
    if bounds is None:
        bounds = [(0.001, 0.999), (-12, 12), (-15, 15)]
    out = []
    for s in range(n_seeds):
        r = differential_evolution(J, bounds, args=(target,), popsize=popsize,
                                    maxiter=maxiter, tol=1e-11, seed=s,
                                    mutation=0.6, recombination=0.5)
        out.append((tuple(r.x), float(r.fun), s, int(r.nfev)))
    return out


def cluster(results, tol=2e-2):
    cl = []
    for theta, j, s, nf in results:
        placed = False
        for c in cl:
            if np.linalg.norm(np.array(theta) - np.array(c[0][0])) < tol:
                c.append((theta, j, s, nf)); placed = True; break
        if not placed:
            cl.append([(theta, j, s, nf)])
    return cl


t0 = time.time()
print("="*72)
print("TEST A: piecewise-linear target θ_true=(0.3, 2.0, -1.5)")
print("="*72)
theta_true = (0.3, 2.0, -1.5)
target_A = gsig(theta_true)
print(f"Target σ: " + ", ".join(f"σ{k}={v:+.4f}" for k, v in target_A.items()))
res_A = run_de(target_A, n_seeds=10)
best_A = min(res_A, key=lambda r: r[1])
err_A = np.linalg.norm(np.array(best_A[0]) - np.array(theta_true))
print(f"DE θ* (best of 10 seeds): ({best_A[0][0]:.7f}, {best_A[0][1]:.7f}, {best_A[0][2]:.7f})")
print(f"θ_true                  : ({theta_true[0]:.7f}, {theta_true[1]:.7f}, {theta_true[2]:.7f})")
print(f"||Δθ|| = {err_A:.3e}, J* = {best_A[1]:.3e}, nfev = {best_A[3]}")
print(f"Test A: {'PASS' if err_A < 1e-3 else 'FAIL'}\n")

print("="*72)
print("TEST B: sin(πt) target; DE vs algebraic Sol-#1")
print("="*72)
target_B = {(1,): 1.0, (2,): 0.0, (1,2): -2/PI, (1,1,2): -1/PI, (1,2,2): 0.25}
sol1 = (0.5, 8/PI, -8/PI)
J_sol1 = J(sol1, target_B)
print(f"Sol-#1 (closed form): θ=({sol1[0]:.5f}, {sol1[1]:.5f}, {sol1[2]:.5f})")
print(f"  J(Sol-#1) = {J_sol1:.3e}  (drops eq#5, residual on σ_(1,2,2))")
res_B = run_de(target_B, n_seeds=15)
best_B = min(res_B, key=lambda r: r[1])
js = sorted([r[1] for r in res_B])
print(f"DE θ* (best of 15 seeds): ({best_B[0][0]:.5f}, {best_B[0][1]:.5f}, {best_B[0][2]:.5f})")
print(f"  J(θ*) = {best_B[1]:.3e}, J spread [{js[0]:.3e}, {js[-1]:.3e}]")
print(f"  ||θ* - Sol-#1|| = {np.linalg.norm(np.array(best_B[0]) - np.array(sol1)):.3e}")
print(f"  J improvement over Sol-#1: {(1 - best_B[1]/J_sol1)*100:.1f}%")

print()
print("="*72)
print("TEST C: sin(πt²)+1-t — paper reports two solutions")
print("="*72)
f_C = lambda t: np.sin(PI*t**2) + 1 - t
T = 1.0
int_f, _ = quad(f_C, 0, T, limit=60)
int_tf, _ = quad(lambda t: t*f_C(t), 0, T, limit=60)
z12 = T*f_C(T) - int_f
z112 = T**2/2*f_C(T) - int_tf
df = lambda t: (f_C(t+1e-5)-f_C(t-1e-5))/2e-5
z122, _ = quad(lambda t3: quad(lambda t2: t2*df(t2), 0, t3, limit=30)[0]*df(t3),
                0, T, limit=30)
target_C = {(1,): T, (2,): f_C(T)-f_C(0), (1,2): z12, (1,1,2): z112, (1,2,2): z122}
print(f"Target σ: " + ", ".join(f"σ{k}={v:+.5f}" for k, v in target_C.items()))

sol1_p = (0.891494, 0.132603, -10.3056)
sol2_p = (0.778296, 0.297333, -5.55432)
print(f"\nPaper Sol-#1: θ={sol1_p}, J={J(sol1_p, target_C):.3e}")
print(f"Paper Sol-#2: θ={sol2_p}, J={J(sol2_p, target_C):.3e}")

res_C = run_de(target_C, n_seeds=25, bounds=[(0.001, 0.999), (-20, 20), (-20, 20)])
clusters = cluster(res_C)
print(f"\nDE 25 seeds → {len(clusters)} cluster(s):")
for i, c in enumerate(sorted(clusters, key=lambda x: -len(x))):
    th = np.array([t for t, _, _, _ in c])
    js = np.array([j for _, j, _, _ in c])
    m = th.mean(axis=0)
    print(f"  C{i+1}: {len(c)}/25 hits, mean θ=({m[0]:+.4f},{m[1]:+.4f},{m[2]:+.4f}), "
          f"J∈[{js.min():.2e},{js.max():.2e}]")

# Strategy (b): seeded near paper Sol-#1 and Sol-#2
print("\nStrategy (b): seed DE population around paper solutions")
rng = np.random.default_rng(0)
for label, sol_p in [("Sol-#1", sol1_p), ("Sol-#2", sol2_p)]:
    init = np.array([[sol_p[0] + 1e-3*rng.standard_normal(),
                      sol_p[1] + 1e-2*rng.standard_normal(),
                      sol_p[2] + 5e-2*rng.standard_normal()] for _ in range(15)])
    r = differential_evolution(J, [(0.001, 0.999), (-20, 20), (-20, 20)],
                                args=(target_C,), popsize=15, maxiter=80,
                                tol=1e-11, init=init, mutation=0.6, recombination=0.5)
    err = np.linalg.norm(r.x - np.array(sol_p))
    print(f"  Seeded {label}: θ*=({r.x[0]:+.4f},{r.x[1]:+.4f},{r.x[2]:+.4f}), "
          f"J={r.fun:.3e}, ||Δ-paper||={err:.3e}")

print(f"\nTotal wall time: {time.time()-t0:.1f}s")
