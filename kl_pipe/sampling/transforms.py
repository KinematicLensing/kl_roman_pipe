"""Smooth bijections from bounded prior supports to unconstrained coordinates.

NUTS on a posterior whose priors have hard truncation bounds sees a -inf
potential wall wherever a trajectory crosses a bound, and every wall crossing
is recorded as a divergence regardless of step size or mass matrix quality.
Sampling in unconstrained coordinates removes the walls exactly: each bounded
parameter is mapped to the real line by a smooth bijection, the potential
gains the log-Jacobian correction, and the posterior in physical coordinates
is unchanged by construction. This is the same treatment numpyro applies to
its native constrained sites; the ``potential_fn`` preconditioned path
bypasses that machinery, so it needs an explicit transform.

The transform per parameter is chosen from the prior's support bounds:

- ``(a, b)`` both finite: affine-logit, ``eta = logit((theta - a) / (b - a))``
- ``(a, None)``: shifted log, ``eta = log(theta - a)``
- ``(None, b)``: reflected log, ``eta = log(b - theta)``
- ``(None, None)``: identity

Periodic parameters (e.g. a position angle on ``[0, 2pi)``) are treated as
plain bounded intervals; that is only appropriate while their posterior mass
stays far from the wrap point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, TYPE_CHECKING

import numpy as np
import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from kl_pipe.priors import PriorDict

# transform kind codes (per-dimension)
_IDENTITY = 0
_LOGIT = 1
_LOG = 2
_NEG_LOG = 3

_KIND_NAMES = {_IDENTITY: 'identity', _LOGIT: 'logit', _LOG: 'log', _NEG_LOG: 'neg_log'}


@dataclass(frozen=True)
class UnconstrainingTransform:
    """Per-dimension bijection eta <-> theta selected from prior bounds.

    ``inverse`` and ``log_jacobian`` are pure jax functions safe inside a
    jitted potential. ``forward`` runs host-side (numpy) and raises loudly on
    values outside the open support; ``forward_clipped`` is the guarded
    variant for a MAP point that may sit exactly on a bound.

    Attributes
    ----------
    kinds : np.ndarray
        Per-dimension transform code (int, static).
    lows, highs : np.ndarray
        Support bounds; entries are only meaningful where the kind uses them
        (placeholders 0/1 elsewhere, chosen so vectorized branches stay
        finite).
    names : tuple of str
        Sampled parameter names (diagnostics only).
    """

    kinds: np.ndarray
    lows: np.ndarray
    highs: np.ndarray
    names: Tuple[str, ...] = field(default=())

    @classmethod
    def from_priors(cls, priors: 'PriorDict') -> 'UnconstrainingTransform':
        """Build the transform from a PriorDict's sampled-parameter bounds."""
        names = tuple(priors.sampled_names)
        kinds = np.empty(len(names), dtype=np.int64)
        lows = np.zeros(len(names))
        highs = np.ones(len(names))
        for i, (low, high) in enumerate(priors.get_bounds()):
            if low is not None and high is not None:
                if high <= low:
                    raise ValueError(
                        f"prior for '{names[i]}' has high ({high}) <= low ({low})"
                    )
                kinds[i] = _LOGIT
                lows[i], highs[i] = low, high
            elif low is not None:
                kinds[i] = _LOG
                lows[i] = low
            elif high is not None:
                kinds[i] = _NEG_LOG
                highs[i] = high
            else:
                kinds[i] = _IDENTITY
        return cls(kinds=kinds, lows=lows, highs=highs, names=names)

    @property
    def kind_names(self) -> Tuple[str, ...]:
        return tuple(_KIND_NAMES[int(k)] for k in self.kinds)

    @property
    def is_identity(self) -> bool:
        return bool((self.kinds == _IDENTITY).all())

    # ---------------------------------------------------------------- forward
    def _forward_np(self, theta: np.ndarray) -> np.ndarray:
        """Vectorized numpy forward map; assumes theta strictly inside support."""
        theta = np.asarray(theta, dtype=np.float64)
        eta = theta.copy()
        m = self.kinds == _LOGIT
        if m.any():
            u = (theta[..., m] - self.lows[m]) / (self.highs[m] - self.lows[m])
            eta[..., m] = np.log(u) - np.log1p(-u)
        m = self.kinds == _LOG
        if m.any():
            eta[..., m] = np.log(theta[..., m] - self.lows[m])
        m = self.kinds == _NEG_LOG
        if m.any():
            eta[..., m] = np.log(self.highs[m] - theta[..., m])
        return eta

    def _support_violations(self, theta: np.ndarray) -> np.ndarray:
        """Boolean mask (broadcast over leading dims) of on/out-of-support dims."""
        theta = np.asarray(theta, dtype=np.float64)
        bad = np.zeros(theta.shape, dtype=bool)
        m = self.kinds == _LOGIT
        bad[..., m] = (theta[..., m] <= self.lows[m]) | (theta[..., m] >= self.highs[m])
        m = self.kinds == _LOG
        bad[..., m] |= theta[..., m] <= self.lows[m]
        m = self.kinds == _NEG_LOG
        bad[..., m] |= theta[..., m] >= self.highs[m]
        return bad

    def forward(self, theta: np.ndarray) -> np.ndarray:
        """Map physical theta to unconstrained eta; raise if on/outside support."""
        bad = self._support_violations(theta)
        if bad.any():
            idx = sorted(set(np.where(bad)[-1].tolist()))
            labels = [self.names[i] if self.names else str(i) for i in idx]
            raise ValueError(
                f"theta on or outside the open prior support for {labels}; "
                f"use forward_clipped for boundary-pinned points"
            )
        return self._forward_np(theta)

    def forward_clipped(
        self, theta: np.ndarray, u_margin: float = 1e-6
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Forward map with boundary guard for points on/near/outside a bound.

        Bounded dims are clipped to at least ``u_margin`` of the interval
        width from each wall; log dims to ``u_margin`` above the bound (in
        units of ``1 + |bound|``). Returns ``(eta, clipped_mask)`` so callers
        can report which dimensions were moved.
        """
        theta = np.asarray(theta, dtype=np.float64).copy()
        clipped = np.zeros(theta.shape, dtype=bool)
        m = self.kinds == _LOGIT
        if m.any():
            width = self.highs[m] - self.lows[m]
            lo = self.lows[m] + u_margin * width
            hi = self.highs[m] - u_margin * width
            t = theta[..., m]
            c = (t < lo) | (t > hi)
            theta[..., m] = np.clip(t, lo, hi)
            clipped[..., m] = c
        for kind, sign_high in ((_LOG, False), (_NEG_LOG, True)):
            m = self.kinds == kind
            if m.any():
                bound = self.highs[m] if sign_high else self.lows[m]
                margin = u_margin * (1.0 + np.abs(bound))
                t = theta[..., m]
                if sign_high:
                    c = t > bound - margin
                    theta[..., m] = np.minimum(t, bound - margin)
                else:
                    c = t < bound + margin
                    theta[..., m] = np.maximum(t, bound + margin)
                clipped[..., m] = c
        return self._forward_np(theta), clipped

    # ---------------------------------------------------------------- inverse
    def inverse(self, eta: jnp.ndarray) -> jnp.ndarray:
        """Map unconstrained eta back to physical theta (jax, trace-safe)."""
        eta = jnp.asarray(eta)
        kinds = jnp.asarray(self.kinds)
        lows = jnp.asarray(self.lows)
        highs = jnp.asarray(self.highs)
        theta_logit = lows + (highs - lows) * jax.nn.sigmoid(eta)
        # exp guarded against overflow only where actually used
        exp_eta = jnp.exp(jnp.where((kinds == _LOG) | (kinds == _NEG_LOG), eta, 0.0))
        theta_log = lows + exp_eta
        theta_neg = highs - exp_eta
        out = jnp.where(kinds == _LOGIT, theta_logit, eta)
        out = jnp.where(kinds == _LOG, theta_log, out)
        out = jnp.where(kinds == _NEG_LOG, theta_neg, out)
        return out

    def log_jacobian(self, eta: jnp.ndarray) -> jnp.ndarray:
        """Sum over dims of log|dtheta/deta| at eta (jax, trace-safe)."""
        eta = jnp.asarray(eta)
        kinds = jnp.asarray(self.kinds)
        widths = jnp.asarray(self.highs - self.lows)
        lj_logit = (
            jnp.log(jnp.where(kinds == _LOGIT, widths, 1.0))
            + jax.nn.log_sigmoid(eta)
            + jax.nn.log_sigmoid(-eta)
        )
        per_dim = jnp.where(kinds == _LOGIT, lj_logit, 0.0)
        per_dim = jnp.where((kinds == _LOG) | (kinds == _NEG_LOG), eta, per_dim)
        return jnp.sum(per_dim, axis=-1)

    def jacobian_diag(self, eta: np.ndarray) -> np.ndarray:
        """Per-dim dtheta/deta at eta, host-side numpy (all entries positive).

        Note ``|dtheta/deta|`` is returned (the neg_log branch has negative
        slope); the mass-matrix similarity transform only needs magnitudes.
        """
        eta = np.asarray(eta, dtype=np.float64)
        diag = np.ones(eta.shape)
        m = self.kinds == _LOGIT
        if m.any():
            sig = 1.0 / (1.0 + np.exp(-eta[..., m]))
            diag[..., m] = (self.highs[m] - self.lows[m]) * sig * (1.0 - sig)
        m = (self.kinds == _LOG) | (self.kinds == _NEG_LOG)
        if m.any():
            diag[..., m] = np.exp(eta[..., m])
        return diag

    def transform_inverse_mass(
        self, inverse_mass: np.ndarray, eta: np.ndarray
    ) -> np.ndarray:
        """Map a physical-space inverse mass matrix (posterior covariance
        estimate) into eta coordinates via the Jacobian at eta:
        ``cov_eta = D^-1 cov_theta D^-1`` with ``D = diag(dtheta/deta)``."""
        d = self.jacobian_diag(eta)
        return np.asarray(inverse_mass) / d[:, None] / d[None, :]
