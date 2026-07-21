"""Dispersion-leg layout variants: BCOO vs dense-gather vs scan-streaming.

Ported from experiments/sweverett/production_speedups/stream_fusion_probe.py
with two changes: (1) the module-level jax_enable_x64 force is REMOVED so
the kit's precision mode (fp64 or fp32 pass) governs; (2) construction is
wrapped in a build function instead of main(). Math and layouts are
otherwise identical. Keep in sync with the source probe.

Isolates the cube-assembly + dispersion legs with fixed (I, lam_obs, sigma)
inputs; gradients flow through all three. Variants:
  A_bcoo         : erf-kernel cube + sparse BCOO matvec (shipped semantics)
  A2_gather      : same cube, dense per-slice gather taps (GPU-friendly layout)
  B_stream       : lax.scan over wavelength, no remat
  C_stream_remat : scan + jax.checkpoint body (O(npix) memory)

CPU verdict (2026-07-04): BCOO wins decisively at our ~99.995% sparsity.
This kit re-runs the matrix on GH200 where cuSPARSE lowering and the
bandwidth/compute ratio may reverse it.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

SQRT2 = np.sqrt(2.0)


def _catmull_rom_weights(f):
    f2 = f * f
    f3 = f2 * f
    w0 = -0.5 * f3 + f2 - 0.5 * f
    w1 = 1.5 * f3 - 2.5 * f2 + 1.0
    w2 = -1.5 * f3 + 2.0 * f2 + 0.5 * f
    w3 = 0.5 * f3 - 0.5 * f2
    return np.stack([w0, w1, w2, w3])


def build_operators(nrow, ncol, nlam, dispersion_pix_per_slice, angle, rot):
    """Return (BCOO, gather_idx (Nlam,16,npix), gather_w (Nlam,16,npix))."""
    from jax.experimental import sparse as jsparse

    npix = nrow * ncol
    rows_grid, cols_grid = np.mgrid[0:nrow, 0:ncol].astype(np.float64)
    c_row, c_col = (nrow - 1) / 2.0, (ncol - 1) / 2.0
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    cos_r, sin_r = np.cos(rot), np.sin(rot)
    dlam = 1.0  # absorbed into weights uniformly; irrelevant for timing

    offsets = (np.arange(nlam) - (nlam - 1) / 2.0) * dispersion_pix_per_slice

    out_rows, in_cols, weights = [], [], []
    g_idx = np.zeros((nlam, 16, npix), dtype=np.int32)
    g_w = np.zeros((nlam, 16, npix), dtype=np.float64)
    row_index = np.arange(npix)

    for k in range(nlam):
        u = cols_grid - offsets[k] * cos_a - c_col
        v = rows_grid - offsets[k] * sin_a - c_row
        row_s = c_row + sin_r * u + cos_r * v
        col_s = c_col + cos_r * u - sin_r * v
        i0 = np.floor(row_s)
        j0 = np.floor(col_s)
        w_row = _catmull_rom_weights(row_s - i0)
        w_col = _catmull_rom_weights(col_s - j0)
        t = 0
        for di in range(4):
            ii = i0.astype(np.int64) + (di - 1)
            valid_i = (ii >= 0) & (ii < nrow)
            for dj in range(4):
                jj = j0.astype(np.int64) + (dj - 1)
                valid = valid_i & (jj >= 0) & (jj < ncol)
                w = (w_row[di] * w_col[dj] * dlam) * valid
                flat_in = (
                    np.clip(ii, 0, nrow - 1) * ncol + np.clip(jj, 0, ncol - 1)
                ).ravel()
                wr = w.ravel()
                g_idx[k, t] = flat_in
                g_w[k, t] = wr
                t += 1
                keep = wr != 0.0
                if keep.any():
                    out_rows.append(row_index[keep])
                    in_cols.append((k * npix + flat_in)[keep])
                    weights.append(wr[keep])

    data = jnp.asarray(np.concatenate(weights))
    coords = jnp.asarray(
        np.stack([np.concatenate(out_rows), np.concatenate(in_cols)], axis=1).astype(
            np.int32
        )
    )
    bcoo = jsparse.BCOO((data, coords), shape=(npix, npix * nlam))
    return bcoo, jnp.asarray(g_idx), jnp.asarray(g_w)


def make_variants(bcoo, g_idx, g_w, edges, dl, shape):
    nrow, ncol = shape
    npix = nrow * ncol
    nlam = g_idx.shape[0]

    def cube_full(I, lam, sig):
        u = (edges[None, None, :] - lam[:, :, None]) / (sig[:, :, None] * SQRT2)
        cdf = 0.5 * (1.0 + jax.scipy.special.erf(u))
        kernel = (cdf[:, :, 1:] - cdf[:, :, :-1]) / dl
        return I[:, :, None] * kernel

    def variant_A(I, lam, sig):
        cube = cube_full(I, lam, sig)
        flat = jnp.transpose(cube, (2, 0, 1)).ravel()
        return (bcoo @ flat).reshape(nrow, ncol)

    def variant_A2(I, lam, sig):
        cube = cube_full(I, lam, sig)  # (nr, nc, nlam)
        flat = jnp.transpose(cube, (2, 0, 1)).reshape(nlam, npix)
        gathered = flat[jnp.arange(nlam)[:, None, None], g_idx]
        out = jnp.sum(g_w * gathered, axis=(0, 1))
        return out.reshape(nrow, ncol)

    def _slice_contrib(k_args, I, lam, sig):
        e_lo, e_hi, idx_k, w_k = k_args
        u_lo = (e_lo - lam) / (sig * SQRT2)
        u_hi = (e_hi - lam) / (sig * SQRT2)
        kern = 0.5 * (jax.scipy.special.erf(u_hi) - jax.scipy.special.erf(u_lo)) / dl
        slice_flat = (I * kern).ravel()
        return jnp.sum(w_k * slice_flat[idx_k], axis=0)  # (npix,)

    def make_stream(remat):
        body = _slice_contrib
        if remat:
            body = jax.checkpoint(_slice_contrib)

        def variant(I, lam, sig):
            def step(acc, k_args):
                return acc + body(k_args, I, lam, sig), None

            init = jnp.zeros(npix)
            xs = (edges[:-1], edges[1:], g_idx, g_w)
            out, _ = jax.lax.scan(step, init, xs)
            return out.reshape(nrow, ncol)

        return variant

    return {
        'A_bcoo': variant_A,
        'A2_gather': variant_A2,
        'B_stream': make_stream(remat=False),
        'C_stream_remat': make_stream(remat=True),
    }


def build_problem(nrow=96, ncol=96, nlam=33, seed=11):
    """Flagship-geometry synthetic inputs (fine grid 96x96, Nlam 33)."""
    rng = np.random.default_rng(seed)
    lam0 = 1313.0
    dl = 1.1
    edges = jnp.asarray(lam0 + dl * (np.arange(nlam + 1) - nlam / 2))
    I = jnp.asarray(rng.random((nrow, ncol)))
    lam = jnp.asarray(lam0 + 2.0 * rng.standard_normal((nrow, ncol)))
    sig = jnp.asarray(0.22 * (1 + 0.3 * rng.random((nrow, ncol))))
    ct = jnp.asarray(rng.standard_normal((nrow, ncol)))
    return edges, dl, I, lam, sig, ct
