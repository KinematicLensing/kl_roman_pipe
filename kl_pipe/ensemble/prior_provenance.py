"""
Per-parameter provenance registry for the catalog-mode ensemble.

One entry per fit parameter (sampled, fixed, or constructor-pinned): what the
parameter means, how its truth is painted, what prior the fit applies, the
provenance class, notes, and the literature keys. Every numeric in the
rendered table is imported from the module that uses it (never copied), so the
table cannot drift from the code; ``tests/test_prior_provenance.py`` asserts
the registry's key set equals the production scene's parameter set.

Distribution shorthand used in the painted/prior cells:
N(mean, sigma) Gaussian; TN truncated Gaussian, support in brackets;
LN(median, scatter) log-normal, scatter in dex unless marked 'ln';
TLN truncated log-normal; CLN log-normal on the ratio to the sampled parent
scale (``ConditionalLogNormal``); U(low, high) uniform.

Classes
-------
catalog fit
    Truth is a catalog column; the prior is a distribution fit to the
    selected sample of that column.
paint
    Truth is drawn from an assumed population distribution (the catalog does
    not carry it); the fit prior is that same generating distribution.
ratio to parent
    Truth is painted as a ratio to the catalog disk scale; the prior
    conditions on the sampled parent scale through the same ratio constants.
mock measurement
    Prior is centered on a simulated per-galaxy measurement (never on the
    per-galaxy truth itself).
instrument scale
    Prior width set by an instrument scale (pixel, dispersion), deliberately
    wider than the paint so the direction stays data-dominated.
pinned
    Fixed at the same value in mock and fit: a stated model assumption, not a
    recovered parameter.
interim
    Deliberate placeholder flagged for replacement (e.g. the bulge Sersic
    index pinned at truth outside production configs).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, Tuple, TYPE_CHECKING

from kl_pipe.ensemble import scene
from kl_pipe.ensemble.catalogs import get_catalog_adapter
from kl_pipe.ensemble.population import (
    BULGE_CLASSICAL_N,
    BULGE_PSEUDO_N,
    BULGE_PSEUDO_WEIGHT,
    BULGE_SIZE_RATIO_LN_SCATTER,
    BULGE_SIZE_RATIO_MAX,
    BULGE_SIZE_RATIO_MEDIAN,
    CENTROID_SCATTER_ARCSEC,
    CONT_CENTROID_OFFSET_ARCSEC,
    HALPHA_RSCALE_RATIO_DEX,
    HALPHA_RSCALE_RATIO_MEDIAN,
    V0_SCATTER_KMS,
    VEL_RSCALE_RATIO_DEX,
    VEL_RSCALE_RATIO_MEDIAN,
)

if TYPE_CHECKING:
    from kl_pipe.ensemble.spec import EnsembleSpec, ObservationConfig


@dataclass(frozen=True)
class PriorProvenance:
    """Provenance record for one fit parameter.

    The compact_* fields override their counterparts in the compact table
    only; empty means fall back to the full-table cell. compact_note is a
    short qualifier column replacing the full notes.
    """

    param: str
    meaning: str
    unit: str
    painted: str
    fit_prior: str
    category: str
    notes: str
    bibkeys: Tuple[str, ...] = field(default_factory=tuple)
    compact_meaning: str = ''
    compact_painted: str = ''
    compact_prior: str = ''
    compact_note: str = ''


_CATEGORIES = (
    'catalog fit',
    'paint',
    'ratio to parent',
    'mock measurement',
    'instrument scale',
    'pinned',
    'interim',
)


def catalog_registry(
    spec: 'EnsembleSpec', config: 'ObservationConfig'
) -> Dict[str, PriorProvenance]:
    """
    Build the provenance registry for a catalog-mode spec.

    Parameters
    ----------
    spec : EnsembleSpec
        Catalog-mode ensemble spec (supplies selection, paint, shear, and
        fit-prior knobs).
    config : ObservationConfig
        Scene structure (bands).

    Returns
    -------
    dict
        Concrete dotted parameter name -> PriorProvenance. Covers every
        sampled and fixed parameter the scene produces, plus the
        constructor-pinned bulge Sersic index when it is not sampled.
    """
    cp = spec.catalog_population
    if cp is None:
        raise ValueError("provenance registry is defined for catalog mode only")
    bulge = cp.paint_bulge

    adapter = get_catalog_adapter(cp.catalog_kind)
    pc = adapter.prior_constants
    cat_keys = adapter.citation_bibkeys
    sig_cont = scene._CONT_CENTROID_SIGMA
    r_lo, r_hi = pc.rscale_low, pc.rscale_high
    size_med = 10.0**pc.rscale_log10_mu
    cont_med = 10.0**pc.cont_flux_log10_mu
    reg: Dict[str, PriorProvenance] = {}

    def add(entry: PriorProvenance) -> None:
        if entry.category not in _CATEGORIES:
            raise ValueError(f"unknown category '{entry.category}' ({entry.param})")
        if entry.param in reg:
            raise ValueError(f"duplicate registry entry '{entry.param}'")
        reg[entry.param] = entry

    # --- shared geometry -----------------------------------------------------
    for g in ('g1', 'g2'):
        add(
            PriorProvenance(
                g,
                f'lensing shear component {g[-1]}',
                '--',
                f'N(0, {cp.shear_sigma}), redrawn to |g| < {cp.shear_gmax}, '
                'pair-shared',
                f'N(0, {spec.shear_fit_prior_sigma})',
                'paint',
                'Wide so the posterior width comes from the data; unbounded '
                'to avoid a prior edge.',
                compact_meaning='shear',
                compact_painted=f'N(0, {cp.shear_sigma})',
                compact_note=f'pair-shared; |g| < {cp.shear_gmax}',
            )
        )
    add(
        PriorProvenance(
            'cosi',
            'disk inclination cos i',
            '--',
            f'U{cp.cosi_range}, isotropic redraw',
            f'U{cp.cosi_range}',
            'paint',
            'The catalog inclination encodes its training-sample selection '
            'and correlates with no retained column, so the redraw is '
            'lossless; the catalog value is kept for validation.',
            compact_meaning='inclination',
            compact_painted=f'U{cp.cosi_range}',
            compact_note='isotropic redraw',
        )
    )
    add(
        PriorProvenance(
            'theta_int',
            'intrinsic position angle',
            'rad',
            'U(0, pi); ring partner at +pi/2',
            'U(0, pi)',
            'paint',
            'Ring partner at +pi/2 averages orientation-dependent residuals '
            'out of the ensemble shear.',
            compact_meaning='intrinsic PA',
            compact_painted='U(0, pi)',
            compact_note='ring partner at +pi/2',
        )
    )
    add(
        PriorProvenance(
            'z',
            'redshift',
            '--',
            'catalog',
            'pinned',
            'pinned',
            (
                'Pinned to the grism spectroscopic value (v1); v0 absorbs '
                'the residual.'
                if adapter.has_observed_redshift
                else 'The painted observables treat the catalog photo-z as '
                'truth; v0 absorbs the residual.'
            ),
            cat_keys,
            compact_note='v0 absorbs the redshift error',
        )
    )

    # --- kinematics ----------------------------------------------------------
    add(
        PriorProvenance(
            'vel.vcirc',
            'asymptotic circular velocity',
            'km/s',
            f'TFR: logv = {cp.tfr_logv0} + (logM - {cp.tfr_logm0})'
            f'/{cp.tfr_slope} + N(0, {cp.tfr_scatter_dex})',
            'LN at TFR(logm_obs); width = TFR scatter plus mass error',
            'mock measurement',
            'Centered on the TFR at a noisy mock photometric mass, '
            f'logm_obs = logM + N(0, {cp.logm_obs_scatter_dex}), so the '
            'center is off-truth by construction. The KL inclination '
            'constraint enters through this prior.',
            ('Ubler2017',),
            compact_meaning='circular velocity',
            compact_painted=f'TFR(logM) + N(0, {cp.tfr_scatter_dex})',
            compact_prior='LN at TFR(logm_obs)',
            compact_note=f'mass error {cp.logm_obs_scatter_dex} dex',
        )
    )
    add(
        PriorProvenance(
            'vel.rscale',
            'arctan turnover radius r_t',
            'arcsec',
            f'R_d x LN({VEL_RSCALE_RATIO_MEDIAN}, {VEL_RSCALE_RATIO_DEX})',
            f'CLN(R_d x {VEL_RSCALE_RATIO_MEDIAN}, {VEL_RSCALE_RATIO_DEX}; '
            f'[{r_lo}, {r_hi}])',
            'ratio to parent',
            'From the public PROBES-I per-galaxy fits (753 late types, '
            'tanh and Courteau-97 refit to arctan); Miller+11 (~0.4 at '
            'z ~ 1) and Catinella+06 (0.24-0.57) agree. The anchor is '
            'z ~ 0; the redshift extrapolation is the stated systematic.',
            ('Stone2022', 'Miller2011', 'Catinella2006'),
            compact_meaning='turnover radius',
            compact_prior='CLN(same)',
            compact_note='z ~ 0 anchor',
        )
    )
    add(
        PriorProvenance(
            'vel.v0',
            'systemic velocity offset',
            'km/s',
            f'N(0, {V0_SCATTER_KMS})',
            f'N(0, {scene._V0_PRIOR_SIGMA_KMS})',
            'instrument scale',
            '3.5x the worst-case (S/N 7) line-centroid precision of 40 '
            'km/s, capping prior-driven shrinkage at <= 30% even for the '
            'noisiest selected galaxy while leaving v0 data-dominated.',
            compact_meaning='systemic velocity',
            compact_note='3.5x worst-case centroid precision',
        )
    )
    add(
        PriorProvenance(
            'Halpha.dispersion',
            'gas velocity dispersion sigma_0',
            'km/s',
            f'N({cp.sigma0_intercept_kms} + {cp.sigma0_slope_kms} z, '
            f'{cp.sigma0_scatter_kms}), floor {cp.sigma0_min_kms}',
            f'TN(same sigma_0(z); [{cp.sigma0_min_kms}, 150])',
            'paint',
            'Prior equals paint: the same affine sigma_0(z) relation and ' 'scatter.',
            ('Ubler2019',),
            compact_meaning='velocity dispersion',
            compact_painted=f'N({cp.sigma0_intercept_kms} + '
            f'{cp.sigma0_slope_kms} z, {cp.sigma0_scatter_kms})',
            compact_prior='TN(same)',
            compact_note=f'floor {cp.sigma0_min_kms} km/s',
        )
    )

    # --- sizes ---------------------------------------------------------------
    size_note = (
        'Population fit to the selected sample. Component sizes are sampled '
        'independently: the right chromatic tie is per-galaxy and unpainted, '
        'and independence widens but cannot bias.'
    )
    band_size_key = '{band}.disk_rscale' if bulge else '{band}.rscale'
    for band in config.bands:
        add(
            PriorProvenance(
                band_size_key.format(band=band),
                f'{band} disk scale length',
                'arcsec',
                'catalog (one size per galaxy, shared with the continuum)',
                f'TLN({size_med:.2f}, {pc.rscale_log10_sigma}; ' f'[{r_lo}, {r_hi}])',
                'catalog fit',
                size_note,
                cat_keys,
                compact_meaning='disk scale length',
                compact_painted='catalog',
                compact_prior=f'TLN({size_med:.2f}, {pc.rscale_log10_sigma})',
                compact_note='shared with the continuum',
            )
        )
    add(
        PriorProvenance(
            'Halpha.rscale',
            'Halpha disk scale length',
            'arcsec',
            f'R_d x LN({HALPHA_RSCALE_RATIO_MEDIAN}, {HALPHA_RSCALE_RATIO_DEX})',
            f'CLN(R_d x {HALPHA_RSCALE_RATIO_MEDIAN}, '
            f'{HALPHA_RSCALE_RATIO_DEX}; [{r_lo}, {r_hi}])',
            'ratio to parent',
            'Line emission extends beyond the continuum (inside-out '
            'growth); median and rms from one sample. Matharu+22 confirm '
            'the median but give no width.',
            ('Nelson2012', 'Matharu2022'),
            compact_meaning='line scale length',
            compact_prior='CLN(same)',
            compact_note='line extends beyond continuum',
        )
    )
    add(
        PriorProvenance(
            'Halpha.cont.rscale',
            'line-continuum disk scale length',
            'arcsec',
            'catalog (same stellar disk as the bands)',
            f'TLN({size_med:.2f}, {pc.rscale_log10_sigma}; ' f'[{r_lo}, {r_hi}])',
            'catalog fit',
            size_note,
            cat_keys + ('vanderWel2014',),
            compact_meaning='continuum scale length',
            compact_painted='catalog',
            compact_prior=f'TLN({size_med:.2f}, {pc.rscale_log10_sigma})',
            compact_note='same stellar disk as the bands',
        )
    )

    # --- amplitudes ----------------------------------------------------------
    flux_note = (
        'Mock photometric measurement: center = truth + one seeded noise '
        'draw at the expected error, width = that same error, both from '
        'the published depth anchor (matched-filter, '
        'compactness-corrected). Deliberately off-truth.'
    )
    add(
        PriorProvenance(
            'Halpha.flux',
            'integrated Halpha line flux',
            '1e-17 erg/s/cm2',
            'painted catalog line flux',
            f'TN(measured, f_lim_coadd/5 * C_ref/C; '
            f'[{pc.line_flux_low}, {pc.line_flux_high}])',
            'mock measurement',
            flux_note,
            cat_keys,
            compact_meaning='line flux',
            compact_painted='catalog',
            compact_prior='TN(measured, expected error)',
            compact_note='center offset by one noise draw',
        )
    )
    for band in config.bands:
        flux_key = f'{band}.total_flux' if bulge else f'{band}.flux'
        add(
            PriorProvenance(
                flux_key,
                f'{band} total broadband flux',
                'uJy',
                'catalog photometry interpolated to the Roman band ' '(line-inclusive)',
                f'TN(measured, f_lim_ps/5 / C; '
                f'[{pc.band_flux_low}, {pc.band_flux_high}])',
                'mock measurement',
                flux_note,
                cat_keys,
                compact_meaning='broadband flux',
                compact_painted='catalog photometry',
                compact_prior='TN(measured, expected error)',
                compact_note='center offset by one noise draw',
            )
        )
    add(
        PriorProvenance(
            'Halpha.cont.flux_per_nm',
            'continuum flux density under the line',
            '1e-17 erg/s/cm2/nm',
            'line flux / observed EW (catalog rest EW)',
            f'TLN({cont_med:.2f}, {pc.cont_flux_log10_sigma}; '
            f'[{pc.cont_flux_low}, {pc.cont_flux_high}])',
            'catalog fit',
            'Population of the selected catalog EWs; no per-galaxy truth, '
            'and tighter than the truth-centered prior it replaced.',
            cat_keys + ('Khostovan2024',),
            compact_meaning='continuum flux density',
            compact_painted='line flux / catalog EW',
            compact_prior=f'TLN({cont_med:.2f}, {pc.cont_flux_log10_sigma})',
            compact_note='population of catalog EWs',
        )
    )

    # --- centroids -----------------------------------------------------------
    for comp in tuple(config.bands) + ('Halpha',):
        for ax in ('x0', 'y0'):
            add(
                PriorProvenance(
                    f'{comp}.{ax}',
                    f'{comp} centroid offset {ax[0]}',
                    'arcsec',
                    f'N(0, {CENTROID_SCATTER_ARCSEC})',
                    f'TN(0, {CENTROID_SCATTER_ARCSEC}; '
                    f'[{scene._CENTROID_BOUNDS[0]}, {scene._CENTROID_BOUNDS[1]}])',
                    'instrument scale',
                    'Per-component astrometric offset, about one Roman pixel.',
                    compact_meaning='centroid offset',
                    compact_prior=f'TN(0, {CENTROID_SCATTER_ARCSEC})',
                    compact_note='about one pixel',
                )
            )
    for ax in ('x0', 'y0'):
        add(
            PriorProvenance(
                f'Halpha.cont.{ax}',
                f'continuum centroid offset {ax[0]}',
                'arcsec',
                f'line centroid + N(0, {CONT_CENTROID_OFFSET_ARCSEC})',
                f'TN(0, {sig_cont:.3f}; '
                f'[{scene._CENTROID_BOUNDS[0]}, {scene._CENTROID_BOUNDS[1]}])',
                'paint',
                'Clumpy star formation offset from the older stellar disk, '
                'half a pixel (~0.45 kpc). Prior width is the two paint '
                'scatters in quadrature, marginal not conditional, so the '
                'offset is measured rather than imposed.',
                ('Nelson2012',),
                compact_meaning='continuum centroid offset',
                compact_painted=f'line + N(0, {CONT_CENTROID_OFFSET_ARCSEC})',
                compact_prior=f'TN(0, {sig_cont:.3f})',
                compact_note='clumpy offset; width = quadrature of the '
                'astrometric floor and this offset, not independently set',
            )
        )

    # --- vertical structure (pinned) -----------------------------------------
    thick = (
        'Direct sech^2 z0/Rd measurements span 0.2-0.38 in this convention; '
        'C/A ellipsoid inversions are upper bounds. A dedicated subset run '
        'samples the thickness to isolate its effect.'
    )
    for comp in ('Halpha', 'Halpha.cont'):
        add(
            PriorProvenance(
                f'{comp}.h_over_r',
                f'{comp} scale height / scale length',
                '--',
                f'{scene._DISK_H_OVER_R}',
                'pinned',
                'pinned',
                thick,
                ('Kregel2002', 'Yu2026', 'vanAsselt2026', 'Hoffmann2022'),
                compact_meaning='disk thickness',
                compact_note='sampled in a dedicated subset run',
            )
        )
    disk_h_key = '{band}.disk_h_over_r' if bulge else '{band}.h_over_r'
    for band in config.bands:
        add(
            PriorProvenance(
                disk_h_key.format(band=band),
                f'{band} disk scale height / scale length',
                '--',
                f'{scene._DISK_H_OVER_R}',
                'pinned',
                'pinned',
                thick,
                ('Kregel2002', 'Yu2026', 'vanAsselt2026', 'Hoffmann2022'),
                compact_meaning='disk thickness',
                compact_note='sampled in a dedicated subset run',
            )
        )
    if bulge:
        for band in config.bands:
            add(
                PriorProvenance(
                    f'{band}.bulge_h_over_hlr',
                    f'{band} bulge scale height / half-light radius',
                    '--',
                    f'{scene._BULGE_H_OVER_HLR}',
                    'pinned',
                    'pinned',
                    'Nearest measured intrinsic bulge C/A: 0.65 (Bertola+91) '
                    'and 0.55 (CALIFA, Costantin+18). The scale-height '
                    'conversion is approximate; a B/T <= 0.3 bulge carries '
                    'little inclination signal.',
                    ('Bertola1991', 'Costantin2018'),
                )
            )

    # --- bulge decomposition -------------------------------------------------
    if bulge:
        for band in config.bands:
            add(
                PriorProvenance(
                    f'{band}.bulge_frac',
                    f'{band} bulge-to-total flux ratio',
                    '--',
                    f'catalog; selection B/T <= {cp.bulge_fraction_max}',
                    f'TN({pc.bulge_frac_loc}, {pc.bulge_frac_scale}; '
                    f'[0, {cp.bulge_fraction_max}])',
                    'catalog fit',
                    'Moments of the selected population; adds curvature '
                    'where an unresolved bulge leaves the likelihood flat, '
                    'without centering on truth.',
                    cat_keys + ('Dimauro2018',),
                )
            )
            add(
                PriorProvenance(
                    f'{band}.bulge_hlr',
                    f'{band} bulge half-light radius',
                    'arcsec',
                    f'disk r50 x LN({BULGE_SIZE_RATIO_MEDIAN}, '
                    f'{BULGE_SIZE_RATIO_LN_SCATTER} ln), '
                    f'ratio < {BULGE_SIZE_RATIO_MAX}',
                    f'CLN(disk scale x '
                    f'{BULGE_SIZE_RATIO_MEDIAN * scene._EXP_R50_OVER_RSCALE:.2f}, '
                    f'{BULGE_SIZE_RATIO_LN_SCATTER} ln; '
                    f'[{pc.bulge_hlr_low}, {pc.bulge_hlr_high}])',
                    'ratio to parent',
                    'The catalog bulge size is an uncorrelated random draw '
                    '(documented catalog limitation), so it is repainted. '
                    'The paint caps the ratio below 1, cutting only the '
                    '3-sigma tail.',
                    ('Lang2014', 'Gadotti2009'),
                )
            )
            n_painted = (
                f'{BULGE_PSEUDO_WEIGHT} TN{BULGE_PSEUDO_N} + '
                f'{1.0 - BULGE_PSEUDO_WEIGHT:.1f} TN{BULGE_CLASSICAL_N}'
            )
            n_note = (
                'The catalog index is an uncorrelated random draw, '
                'repainted as the pseudo/classical mixture split at n = 2.'
            )
            if spec.sample_bulge_nsersic:
                add(
                    PriorProvenance(
                        f'{band}.bulge_n_sersic',
                        f'{band} bulge Sersic index',
                        '--',
                        n_painted,
                        'same mixture',
                        'paint',
                        n_note + ' Prior equals paint, so marginalization '
                        'is exact; an unresolved bulge returns its prior.',
                        ('FisherDrory2008', 'Gadotti2009', 'MendezAbreu2010'),
                    )
                )
            else:
                add(
                    PriorProvenance(
                        f'{band}.bulge_n_sersic',
                        f'{band} bulge Sersic index',
                        '--',
                        n_painted,
                        'pinned at the per-galaxy painted truth (truth leak)',
                        'interim',
                        n_note + ' This configuration leaks the per-galaxy '
                        'truth; production must enable '
                        'fit.sample_bulge_nsersic.',
                        ('FisherDrory2008', 'Gadotti2009', 'MendezAbreu2010'),
                    )
                )

    return reg


# short human-readable labels for the standalone (non-BibTeX) rendering
_CITE_LABELS = {
    'Castander2025': 'Castander+25 (Flagship2)',
    'Shuntov2025': 'Shuntov+25 (COSMOS-Web)',
    'Ubler2017': 'Übler+17',
    'Ubler2019': 'Übler+19',
    'Miller2011': 'Miller+11',
    'Courteau1997': 'Courteau 97',
    'Catinella2006': 'Catinella+06',
    'Stone2022': 'Stone+22 (PROBES-I)',
    'Nelson2012': 'Nelson+12',
    'Matharu2022': 'Matharu+22',
    'vanderWel2014': 'van der Wel+14',
    'Hoffmann2022': 'Hoffmann+22',
    'Kregel2002': 'Kregel+02',
    'Yu2026': 'Yu+26',
    'vanAsselt2026': 'van Asselt+26',
    'Bertola1991': 'Bertola+91',
    'Costantin2018': 'Costantin+18',
    'Lang2014': 'Lang+14',
    'Gadotti2009': 'Gadotti 09',
    'FisherDrory2008': 'Fisher & Drory 08',
    'MendezAbreu2010': 'Méndez-Abreu+10',
    'Dimauro2018': 'Dimauro+18',
    'Khostovan2024': 'Khostovan+24',
}

_BLOCK_ORDER = (
    'Geometry and shear',
    'Kinematics',
    'Sizes',
    'Amplitudes',
    'Centroids',
    'Vertical structure',
    'Bulge',
)


def _block_of(param: str) -> str:
    if param in ('g1', 'g2', 'cosi', 'theta_int'):
        return 'Geometry and shear'
    if param in ('vel.vcirc', 'vel.rscale', 'vel.v0', 'Halpha.dispersion', 'z'):
        return 'Kinematics'
    if param.endswith(('x0', 'y0')):
        return 'Centroids'
    if 'h_over' in param:
        return 'Vertical structure'
    if 'bulge_frac' in param or 'bulge_n_sersic' in param:
        return 'Bulge'
    if 'flux' in param:
        return 'Amplitudes'
    if 'rscale' in param or 'bulge_hlr' in param:
        return 'Sizes'
    raise ValueError(f"no table block for parameter '{param}'")


def _tex_escape(s: str) -> str:
    return (
        s.replace('&', r'\&')
        .replace('%', r'\%')
        .replace('_', r'\_')
        .replace('#', r'\#')
        .replace('<=', r'$\leq$')
        .replace('<', r'$<$')
        .replace('>', r'$>$')
        .replace('sigma_0', r'$\sigma_0$')
        .replace('sech^2', r'sech$^2$')
        .replace('~', r'$\sim$')
    )


def registry_to_latex(
    registry: Dict[str, PriorProvenance],
    mode: str = 'standalone',
    cite_urls: Dict[str, str] = None,
) -> str:
    """
    Render the registry as a LaTeX longtable.

    Parameters
    ----------
    registry : dict
        Output of `catalog_registry`.
    mode : str
        'standalone' renders citations as short human-readable labels;
        'paper' renders them as ``\\citet`` commands for the paper build.
    cite_urls : dict, optional
        Bibkey -> URL. In standalone mode, labels with an entry become
        hyperlinks (requires hyperref in the preamble; the standalone
        document loads it). Ignored in paper mode, where the paper's own
        bibliography handles linking.
    """
    if mode not in ('standalone', 'paper'):
        raise ValueError(f"unknown mode '{mode}'")
    urls = cite_urls or {}

    def one_cite(key: str) -> str:
        label = _tex_escape(_CITE_LABELS.get(key, key))
        url = urls.get(key)
        if url is None:
            return label
        return r'\href{' + url + '}{' + label + '}'

    def cite(keys: Tuple[str, ...]) -> str:
        if not keys:
            return '--'
        if mode == 'paper':
            return r'\citet{' + ','.join(keys) + '}'
        return '; '.join(one_cite(k) for k in keys)

    lines = []
    if mode == 'paper':
        # aastex longtables must be preceded by \startlongtable, else the
        # class's table counter throws "Extra \or" errors
        lines.append(r'\startlongtable')
    lines += [
        r'\begin{longtable}{lll ll l l p{12.5cm}}',
        r'\toprule',
        r'Parameter & Meaning & Unit & Painted truth & Fit prior & Class & '
        r'Reference & Notes \\',
        r'\midrule',
        r'\endhead',
    ]
    order = {name: i for i, name in enumerate(_BLOCK_ORDER)}
    grouped: Dict[str, list] = {}
    for param, e in registry.items():
        grouped.setdefault(_block_of(param), []).append(e)
    for block in sorted(grouped, key=order.__getitem__):
        lines.append(r'\midrule\multicolumn{8}{l}{\textbf{' + block + r'}}\\\midrule')
        for e in sorted(grouped[block], key=lambda e: e.param):
            row = ' & '.join(
                (
                    r'\texttt{' + _tex_escape(e.param) + '}',
                    _tex_escape(e.meaning),
                    _tex_escape(e.unit),
                    _tex_escape(e.painted),
                    _tex_escape(e.fit_prior),
                    _tex_escape(e.category),
                    cite(e.bibkeys),
                    _tex_escape(e.notes),
                )
            )
            lines.append(row + r' \\')
    lines += [r'\bottomrule', r'\end{longtable}']
    return '\n'.join(lines)


def registry_to_compact_latex(
    registry: Dict[str, PriorProvenance],
    bands: Tuple[str, ...],
    mode: str = 'standalone',
    cite_urls: Dict[str, str] = None,
) -> str:
    """
    Compact rendering: per-band and x/y rows merged, one line per row, with
    a short qualifier column in place of the full notes.

    Prototype for the in-paper table; the full `registry_to_latex` table is
    the internal reference. Entries' compact_* fields override the full
    cells; rows merge when every rendered cell but the parameter name
    matches once the band token is generalized.
    """
    if mode not in ('standalone', 'paper'):
        raise ValueError(f"unknown mode '{mode}'")
    urls = cite_urls or {}

    def one_cite(key: str) -> str:
        label = re.sub(r'\s*\(.*\)$', '', _CITE_LABELS.get(key, key))
        label = _tex_escape(label)
        url = urls.get(key)
        if url is None:
            return label
        return r'\href{' + url + '}{' + label + '}'

    def cite(keys: Tuple[str, ...]) -> str:
        if not keys:
            return '--'
        if mode == 'paper':
            return r'\citet{' + ','.join(keys) + '}'
        return '; '.join(one_cite(k) for k in keys)

    def canon(text: str) -> str:
        for b in bands:
            text = text.replace(b, r'\{band\}')
        return text

    groups: Dict[tuple, dict] = {}
    for e in registry.values():
        param = re.sub(r'\.(x0|y0)$', '.x0/y0', canon(e.param))
        meaning = e.compact_meaning or re.sub(
            r' offset [xy]$', ' offset', canon(e.meaning)
        )
        painted = canon(e.compact_painted or e.painted)
        prior = canon(e.compact_prior or e.fit_prior)
        note = e.compact_note or '--'
        key = (e.unit, painted, prior, e.category, e.bibkeys, note)
        g = groups.setdefault(
            key,
            {
                'params': [],
                'meaning': meaning,
                'painted': painted,
                'prior': prior,
                'note': note,
                'first': e,
            },
        )
        if param not in g['params']:
            g['params'].append(param)

    lines = []
    if mode == 'paper':
        lines.append(r'\startlongtable')
    lines += [
        r'\begin{longtable}{l l l l l l l l}',
        r'\toprule',
        r'Parameter & Meaning & Unit & Painted truth & Fit prior & Class & '
        r'Reference & Notes \\',
        r'\midrule',
        r'\endhead',
    ]
    order = {name: i for i, name in enumerate(_BLOCK_ORDER)}
    by_block: Dict[str, list] = {}
    for g in groups.values():
        by_block.setdefault(_block_of(g['first'].param), []).append(g)
    for block in sorted(by_block, key=order.__getitem__):
        lines.append(r'\midrule\multicolumn{8}{l}{\textbf{' + block + r'}}\\\midrule')
        for g in sorted(by_block[block], key=lambda g: g['params'][0]):
            e = g['first']
            suffix = g['params'][0].split('.')[-1]
            if len(g['params']) >= 3 and all(
                p.split('.')[-1] == suffix for p in g['params']
            ):
                param_cell = r'\texttt{' + _tex_escape(suffix) + '} (all components)'
            else:
                param_cell = r'\texttt{' + _tex_escape(', '.join(g['params'])) + '}'
            row = ' & '.join(
                (
                    param_cell,
                    _tex_escape(g['meaning']),
                    _tex_escape(e.unit),
                    _tex_escape(g['painted']),
                    _tex_escape(g['prior']),
                    _tex_escape(e.category),
                    cite(e.bibkeys),
                    _tex_escape(g['note']),
                )
            )
            lines.append(row + r' \\')
    lines += [r'\bottomrule', r'\end{longtable}']
    return '\n'.join(lines)


_STANDALONE_PREAMBLE = r"""\documentclass[9pt]{extarticle}
\usepackage[paperwidth=62cm,paperheight=54cm,margin=1.2cm]{geometry}
\usepackage{longtable,booktabs}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[colorlinks=true,urlcolor=blue]{hyperref}
\setlength{\tabcolsep}{5pt}
\renewcommand{\arraystretch}{1.3}
\pagestyle{empty}
\begin{document}
\section*{Model parameters and priors}
Generated from \texttt{kl\_pipe.ensemble.prior\_provenance}; every numeric is
imported from the pipeline constants.

\medskip
\noindent\textbf{Distributions.} N(mean, sigma) = Gaussian;
TN = truncated Gaussian, support in brackets;
LN(median, scatter) = log-normal, scatter in dex;
TLN = truncated log-normal;
CLN = log-normal on the ratio to the sampled parent scale;
U(low, high) = uniform.
$R_d$ is the exponential disk scale length; in ratio priors it means the
galaxy's fitted disk scale.

\smallskip
\noindent\textbf{Classes.}
\emph{catalog fit}: truth is a catalog column, prior fit to the selected
sample;
\emph{paint}: truth drawn from an assumed population distribution, prior
equal to it;
\emph{ratio to parent}: truth painted as a ratio to the disk scale, prior
conditions on the sampled scale;
\emph{mock measurement}: prior centered on a simulated per-galaxy
measurement;
\emph{instrument scale}: prior width set by a pixel or dispersion scale;
\emph{pinned}: same fixed value in mock and fit;
\emph{interim}: deliberate placeholder flagged for replacement.
"""


def standalone_document(
    registry: Dict[str, PriorProvenance], cite_urls: Dict[str, str] = None
) -> str:
    """Full standalone LaTeX document for PDF rendering outside the paper."""
    return (
        _STANDALONE_PREAMBLE
        + registry_to_latex(registry, mode='standalone', cite_urls=cite_urls)
        + '\n\\end{document}\n'
    )


_COMPACT_PREAMBLE = r"""\documentclass[10pt]{article}
\usepackage[paperwidth=42cm,paperheight=20cm,margin=1.2cm]{geometry}
\usepackage{longtable,booktabs}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[colorlinks=true,urlcolor=blue]{hyperref}
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.25}
\footnotesize
\pagestyle{empty}
\begin{document}
\section*{Model parameters and priors}
N(mean, sigma) = Gaussian; TN = truncated Gaussian; LN(median, scatter in
dex) = log-normal; TLN = truncated log-normal; CLN = log-normal on the
ratio to the fitted disk scale $R_d$; U(low, high) = uniform.
"""


def standalone_compact_document(
    registry: Dict[str, PriorProvenance],
    bands: Tuple[str, ...],
    cite_urls: Dict[str, str] = None,
) -> str:
    """Compact standalone LaTeX document (merged rows, no notes)."""
    url = (cite_urls or {}).get('Shuntov2025')
    cosmos25_cite = (
        r'\href{' + url + '}{Shuntov et al. 2025}' if url else 'Shuntov et al. 2025'
    )
    catalog_note = (
        '\n\\par\\noindent Catalog columns (redshift, mass, size, photometry) are '
        f'COSMOS25 ({cosmos25_cite}); '
        r"H$\alpha$ line flux and EW are painted per Jiachuan's recipe."
        '\n\n'
    )
    return (
        _COMPACT_PREAMBLE
        + catalog_note
        + registry_to_compact_latex(
            registry, bands, mode='standalone', cite_urls=cite_urls
        )
        + '\n\\end{document}\n'
    )
