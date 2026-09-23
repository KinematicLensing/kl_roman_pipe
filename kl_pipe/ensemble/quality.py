"""
Per-fit quality columns derived at fit time or from the summary row.

Three groups, all cheap and auditable from columns that already exist:

- posterior interval columns ``post.<p>.q025 / q16 / q84 / q975`` from the
  pooled draws, and the simulation-based-calibration rank
  ``truth_rank.<p>`` (fraction of draws below the truth) when the truth is
  known;
- failure flags ``flag_gate``, ``flag_map_dev``, ``flag_chi2_excess``,
  ``flag_rotation_ambiguous`` and their count ``n_flags``, with the
  thresholds in one place (``QualityThresholds``);
- the number of posterior draws a fit actually took (``draws_per_fit``), for
  per-draw cost columns in the report.

Periodic parameters (``PERIODIC_PARAMS``) are wrapped about the posterior
median before quantiles are taken and about the truth before ranks, so a
posterior straddling the branch cut reads the same as one that does not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

# parameters whose values live on a circle, with their period. theta_int is a
# full-turn position angle: theta and theta + pi are different rotation
# directions, so the period is 2 pi.
PERIODIC_PARAMS: Dict[str, float] = {'theta_int': 2.0 * np.pi}

INTERVAL_QUANTILES = (
    ('q025', 0.025),
    ('q16', 0.16),
    ('q84', 0.84),
    ('q975', 0.975),
)

FLAG_COLUMNS = (
    'flag_gate',
    'flag_map_dev',
    'flag_chi2_excess',
    'flag_rotation_ambiguous',
)


@dataclass(frozen=True)
class QualityThresholds:
    """Thresholds behind the failure flags.

    ``map_dev_max``: healthy fits sit at ``map_postmean_max_dev`` 0.5-2.7
    over 64 bank fits, a MAP left in a wrong basin at 11-25; 5 splits them.
    ``chi2_excess_k``: a chi-square with ``n_data`` degrees of freedom has
    standard deviation ``sqrt(2 n_data)``; the posterior-mean chi-square of a
    fit stuck in a wrong basin exceeds ``n_data`` by hundreds to thousands,
    so ``k = 5`` standard deviations flags those and no healthy fit.
    ``rotation_margin_nats``: the MAP's log-posterior margin over the best
    counter-rotating start below which the two rotation directions are
    competing modes rather than a mode and a decoy.
    """

    map_dev_max: float = 5.0
    chi2_excess_k: float = 5.0
    rotation_margin_nats: float = 3.0


DEFAULT_THRESHOLDS = QualityThresholds()


def wrap_about(values: np.ndarray, center: float, period: float) -> np.ndarray:
    """Map ``values`` onto the branch ``(center - period/2, center + period/2]``."""
    delta = np.asarray(values, dtype=float) - center
    return center + delta - period * np.round(delta / period)


def circular_mean(values: np.ndarray, period: float) -> float:
    """Mean direction of ``values`` on a circle of the given period."""
    phase = 2.0 * np.pi * np.asarray(values, dtype=float) / period
    angle = np.arctan2(np.mean(np.sin(phase)), np.mean(np.cos(phase)))
    return float(angle * period / (2.0 * np.pi))


def posterior_interval_columns(
    samples: np.ndarray,
    sampled_names: Sequence[str],
    truth: Optional[Mapping[str, float]] = None,
    periods: Mapping[str, float] = PERIODIC_PARAMS,
) -> Dict[str, float]:
    """Quantile and truth-rank columns from pooled posterior draws.

    Parameters
    ----------
    samples : ndarray, shape (n_draws, n_params)
        Pooled draws in ``sampled_names`` order.
    sampled_names : sequence of str
    truth : mapping, optional
        Truth values by parameter name; ``truth_rank.<p>`` is written for
        every sampled parameter present in it.
    periods : mapping
        Period per periodic parameter (default ``PERIODIC_PARAMS``).
    """
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != len(sampled_names):
        raise ValueError(
            f"samples must be (n_draws, {len(sampled_names)}), got {samples.shape}"
        )
    if samples.shape[0] < 2:
        raise ValueError("need at least two draws for posterior intervals")
    out: Dict[str, float] = {}
    for i, name in enumerate(sampled_names):
        draws = samples[:, i]
        period = periods.get(name)
        if period is not None:
            # one contiguous branch: first about the circular mean (safe when
            # the draws straddle the cut), then about the median of that
            # branch, shifted so the median lies in [0, period)
            draws = wrap_about(draws, circular_mean(draws, period), period)
            median = float(np.median(draws))
            draws = wrap_about(draws, median, period) + (
                np.mod(median, period) - median
            )
        for label, q in INTERVAL_QUANTILES:
            out[f'post.{name}.{label}'] = float(np.quantile(draws, q))
        if truth is not None and name in truth:
            t = float(truth[name])
            resid = draws - t
            if period is not None:
                resid = resid - period * np.round(resid / period)
            out[f'truth_rank.{name}'] = float(np.mean(resid < 0.0))
    return out


def _finite(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def quality_flags(
    summary: Mapping[str, object],
    rhat_max: float,
    ess_min: float,
    thresholds: QualityThresholds = DEFAULT_THRESHOLDS,
) -> Dict[str, object]:
    """Failure flags for one summary row (final attempt values).

    A flag whose input column is missing or non-finite is False: the flags
    report evidence of failure, and absent evidence is recorded by the
    absent column, not by a flag.
    """
    flags: Dict[str, object] = {}
    flags['flag_gate'] = bool(
        (_finite(summary.get('max_rhat')) and float(summary['max_rhat']) > rhat_max)
        or (_finite(summary.get('min_ess')) and float(summary['min_ess']) < ess_min)
    )
    dev = summary.get('map_postmean_max_dev')
    flags['flag_map_dev'] = bool(_finite(dev) and float(dev) > thresholds.map_dev_max)
    chi2, n_data = summary.get('postmean_chi2'), summary.get('n_data')
    flags['flag_chi2_excess'] = bool(
        _finite(chi2)
        and _finite(n_data)
        and float(n_data) > 0
        and float(chi2) - float(n_data)
        > thresholds.chi2_excess_k * np.sqrt(2.0 * float(n_data))
    )
    margin = summary.get('map_pa_flip_margin')
    flags['flag_rotation_ambiguous'] = bool(
        _finite(margin) and float(margin) < thresholds.rotation_margin_nats
    )
    flags['n_flags'] = int(sum(flags[c] for c in FLAG_COLUMNS))
    return flags


def draws_per_fit(table: pd.DataFrame, spec) -> pd.Series:
    """Posterior draws each fit took, from the spec and the escalation columns.

    First attempt ``n_chains * n_samples``; a restart escalation replaces it
    by the escalation sample count; a continuation adds
    ``escalation_n_blocks * continue_block`` draws per chain.
    """
    esc = spec.escalation
    base = float(spec.n_chains * spec.n_samples)
    draws = pd.Series(base, index=table.index, dtype=float)
    if 'escalation_mode' not in table.columns:
        return draws
    mode = table['escalation_mode'].fillna('').astype(str)
    draws[mode == 'restart'] = float(spec.n_chains * esc.n_samples)
    if 'escalation_n_blocks' in table.columns:
        blocks = table['escalation_n_blocks'].fillna(0).astype(float)
        cont = mode == 'continue'
        draws[cont] = base + blocks[cont] * float(spec.n_chains * esc.continue_block)
    return draws
