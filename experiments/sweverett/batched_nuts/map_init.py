"""Per-galaxy MAP + Laplace metric through the shared batched program.

Mirrors InferenceTask.laplace_preconditioner (multi-start scaled L-BFGS-B
from prior draws, best-of-finite mode selection, central-difference Hessian
of the negative log-posterior, scale-aware eigenvalue floor) but evaluates
through the ONE compiled shared posterior with the galaxy's dynamic pytree
as data, so nothing recompiles per fit. Truth-free, like production.
"""

import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize


def _fd_hessian_from_grad(vg, theta, scale, bounds, rel_step):
    """Central-difference Hessian of -log posterior from the gradient.

    Steps are prior-scaled and shrunk to keep both stencil points strictly
    inside prior bounds (out-of-support gradients are meaningless through
    the -inf prior barrier). Requires float64.
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError('fd hessian requires float64 (jax_enable_x64)')
    th = np.asarray(theta, dtype=np.float64)
    cols = []
    for j in range(th.size):
        h = rel_step * scale[j]
        low, high = bounds[j]
        if low is not None and np.isfinite(low):
            h = min(h, 0.5 * (th[j] - low))
        if high is not None and np.isfinite(high):
            h = min(h, 0.5 * (high - th[j]))
        if not h > 0:
            raise RuntimeError(
                f'MAP sits at a prior bound for parameter index {j} '
                f'(theta={th[j]:.6g}, bounds=({low}, {high}))'
            )
        tp = th.copy()
        tp[j] += h
        tm = th.copy()
        tm[j] -= h
        _, gp = vg(jnp.asarray(tp))
        _, gm = vg(jnp.asarray(tm))
        gp = np.asarray(gp, dtype=np.float64)
        gm = np.asarray(gm, dtype=np.float64)
        if not (np.all(np.isfinite(gp)) and np.all(np.isfinite(gm))):
            raise RuntimeError(
                f'non-finite gradient at fd stencil for parameter index {j} '
                f'(step {h:.3g})'
            )
        # vg returns grad of +log posterior; difference gives -logpost Hessian
        cols.append((gm - gp) / (2.0 * h))
    H = np.stack(cols, axis=1)
    return 0.5 * (H + H.T)


def laplace_for_fit(
    vg,
    priors,
    *,
    n_starts: int = 4,
    eig_floor: float = 1e-4,
    maxiter: int = 2000,
    seed: int = 0,
    fd_rel_step: float = 1e-5,
):
    """MAP point + regularized inverse-Hessian mass matrix for one galaxy.

    vg maps theta -> (log_posterior, grad) through the shared program with
    this galaxy's dynamic pytree closed over. Returns
    (theta_map, inv_mass, n_starts_converged, condition_number), matching
    the numbers InferenceTask.laplace_preconditioner would produce for the
    same posterior.
    """
    names = list(priors.sampled_names)
    prior_batch = np.asarray(priors.sample(jax.random.PRNGKey(seed), n_samples=512))
    loc = prior_batch.mean(axis=0)
    scale = prior_batch.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)

    def neg_u(u):
        v, g = vg(jnp.asarray(loc + scale * u))
        return float(-v), np.asarray(-g, dtype=np.float64) * scale

    starts = np.asarray(priors.sample(jax.random.PRNGKey(seed + 1), n_samples=n_starts))
    # best finite objective across starts, converged or not: a sharp mode
    # can exceed maxiter yet still be the closest point to the optimum
    best = None
    n_converged = 0
    for s0 in starts:
        u0 = (np.asarray(s0, dtype=np.float64) - loc) / scale
        res = minimize(
            neg_u, u0, jac=True, method='L-BFGS-B', options={'maxiter': maxiter}
        )
        if not np.isfinite(res.fun):
            continue
        if res.success:
            n_converged += 1
        if best is None or res.fun < best.fun:
            best = res
    if best is None:
        raise RuntimeError('no MAP start reached a finite log-posterior')

    theta_map = loc + scale * best.x
    bounds = [priors.get_prior(n).bounds for n in names]
    H = _fd_hessian_from_grad(vg, theta_map, scale, bounds, fd_rel_step)
    # scale-aware regularization: floor eigenvalues of the prior-scale-
    # normalized Hessian so only genuine degeneracy is capped, not the
    # benign physical scale spread
    Hn = (scale[:, None] * H) * scale[None, :]
    Hn = 0.5 * (Hn + Hn.T)
    w, V = np.linalg.eigh(Hn)
    w_floored = np.maximum(w, w.max() * eig_floor)
    inv_n = np.linalg.inv((V * w_floored) @ V.T)
    inv_mass = (scale[:, None] * inv_n) * scale[None, :]
    inv_mass = 0.5 * (inv_mass + inv_mass.T)
    cond = float(w_floored.max() / w_floored.min())
    return theta_map, inv_mass, n_converged, cond
