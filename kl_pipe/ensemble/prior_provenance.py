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
    Truth is a Flagship2 catalog column; the prior is a distribution fit to
    the selected sample of that column.
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
    Deliberate v1 simplification awaiting replacement (e.g. scene-constant
    fluxes pending physical units).
"""

from __future__ import annotations

import math
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
    """Provenance record for one fit parameter."""

    param: str
    meaning: str
    unit: str
    painted: str
    fit_prior: str
    category: str
    notes: str
    bibkeys: Tuple[str, ...] = field(default_factory=tuple)


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
                'Wide so posterior widths reflect the data constraint; left '
                'unbounded because a truncation would reintroduce a prior '
                'edge.',
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
            'The catalog inclination encodes the morphological selection of '
            'its training sample and is measured uncorrelated with every '
            'retained column, so the redraw is lossless; the catalog value '
            'is kept for validation.',
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
            'Ring pairing cancels intrinsic-shape noise at leading order.',
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
                'Pinned to the grism spectroscopic value (v1); the systemic '
                'velocity absorbs the residual.'
                if adapter.has_observed_redshift
                else 'Pinned to the catalog photometric redshift, which the '
                'painted observables treat as truth; the systemic velocity '
                'absorbs the residual.'
            ),
            cat_keys,
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
            'Centered on the Tully-Fisher relation at a noisy mock '
            f'photometric mass, logm_obs = logM + '
            f'N(0, {cp.logm_obs_scatter_dex}), so the center is offset from '
            'truth by construction. The KL inclination constraint enters '
            'through this prior.',
            ('Ubler2017',),
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
            'Derived from the public PROBES-I per-galaxy fits: 753 '
            'late-type galaxies, tanh and Courteau-97 fits refit to the '
            'arctan form. Miller+11 (z ~ 1, "typical 0.4") and Catinella+06 '
            '(0.24-0.57) are consistent. The anchor is z ~ 0; the redshift '
            'extrapolation is the stated systematic.',
            ('Stone2022', 'Miller2011', 'Catinella2006'),
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
            'One grism dispersion pixel, 8x the paint scatter; the line '
            'centroid is measured to 14-40 km/s across the sample, so the '
            'data dominate.',
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
            'Same sigma_0(z) relation and scatter as the paint; affine '
            'redshift evolution.',
            ('Ubler2019',),
        )
    )

    # --- sizes ---------------------------------------------------------------
    size_note = (
        'Population fit to the selected sample. Component sizes are sampled '
        'independently because the correct chromatic tie coefficient is '
        'per-galaxy and unpainted; independence can widen the posterior but '
        'not bias it.'
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
            'Line emission is more extended than the continuum (inside-out '
            'growth); median and rms from a single sample. Matharu+22 '
            'confirm the median but publish no width.',
            ('Nelson2012', 'Matharu2022'),
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
        )
    )

    # --- amplitudes ----------------------------------------------------------
    flux_note = (
        'Simulated photometric measurement: the prior center is the truth '
        'plus one seeded noise draw at the expected measurement error, and '
        'the width is that same error, both derived from the published '
        'depth anchor (matched-filter, compactness-corrected). The center '
        'is deliberately NOT the truth.'
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
            )
        )
    add(
        PriorProvenance(
            'Halpha.cont.flux_per_nm',
            'continuum flux density under the line',
            '1e-17 erg/s/cm2 per nm',
            'line flux / observed EW (catalog rest EW)',
            f'TLN({cont_med:.2f}, {pc.cont_flux_log10_sigma}; '
            f'[{pc.cont_flux_low}, {pc.cont_flux_high}])',
            'catalog fit',
            'Population of the selected catalog equivalent widths; contains '
            'no per-galaxy truth and is tighter than the truth-centered '
            'prior it replaced.',
            cat_keys + ('Khostovan2024',),
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
                'Physical offset of clumpy star formation from the older '
                'stellar disk, half a pixel (about 0.45 kpc). The prior '
                'width is the two paint scatters in quadrature, marginal '
                'rather than conditional, so the line-continuum offset is '
                'measured rather than imposed.',
                ('Nelson2012',),
            )
        )

    # --- vertical structure (pinned) -----------------------------------------
    thick = (
        'Image-simulation standard. The measured z < 1 disk thickness '
        '(intrinsic vertical-to-major axis ratio C/A of about 0.24) implies '
        'about 0.42, which would cost 14% of the apparent-shape lever (38% '
        'edge-on); carried as an open systematic.'
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
                ('Hoffmann2022',),
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
                ('Hoffmann2022',),
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
                    'Nearest measurements of the intrinsic bulge axis ratio '
                    'C/A: 0.65 (Bertola+91) and 0.55 (CALIFA, Costantin+18). '
                    'The conversion to a scale-height ratio is approximate, '
                    'and a B/T <= 0.3 bulge carries little inclination '
                    'signal.',
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
                    'Moments of the selected population. An informative '
                    'prior adds curvature where an unresolved bulge leaves '
                    'the likelihood flat, without centering on truth.',
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
                    '(a documented catalog limitation) and is repainted. The '
                    'paint caps the ratio below 1; the cap removes only the '
                    '3-sigma tail of the prior.',
                    ('Lang2014', 'Gadotti2009'),
                )
            )
            n_painted = (
                f'{BULGE_PSEUDO_WEIGHT} TN{BULGE_PSEUDO_N} + '
                f'{1.0 - BULGE_PSEUDO_WEIGHT:.1f} TN{BULGE_CLASSICAL_N}'
            )
            n_note = (
                'The catalog index is an uncorrelated random draw and is '
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
                        n_note + ' Prior equals paint, so the '
                        'marginalization is exact; an unresolved bulge '
                        'returns its prior.',
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
    if param in ('g1', 'g2', 'cosi', 'theta_int', 'z'):
        return 'Geometry and shear'
    if param in ('vel.vcirc', 'vel.rscale', 'vel.v0', 'Halpha.dispersion'):
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

    lines = [
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
\section*{Catalog-mode ensemble: paint and prior provenance}
Generated from \texttt{kl\_pipe.ensemble.prior\_provenance}; every numeric is
imported from the pipeline constants.

\medskip
\noindent\textbf{Distributions.} N(mean, sigma) = Gaussian;
TN = truncated Gaussian, support in brackets;
LN(median, scatter) = log-normal, scatter in dex unless marked `ln';
TLN = truncated log-normal;
CLN = log-normal on the ratio to the sampled parent scale;
U(low, high) = uniform.

\smallskip
\noindent\textbf{Notation.} $R_d$ is the exponential disk scale length: the
catalog value in the painted truth, and the galaxy's sampled reference-band
disk scale in the fit priors (so a CLN prior conditions on a parameter that
is marginalized with the rest, not on a measurement).

\smallskip
\noindent\textbf{Classes.}
\emph{catalog fit}: truth is a Flagship2 column, prior fit to the selected
sample;
\emph{paint}: truth drawn from an assumed population distribution, prior
equal to it;
\emph{ratio to parent}: truth painted as a ratio to the disk scale, prior
conditions on the sampled scale;
\emph{mock measurement}: prior centered on a simulated per-galaxy
measurement;
\emph{instrument scale}: prior width set by a pixel or dispersion scale;
\emph{pinned}: same fixed value in mock and fit;
\emph{interim}: placeholder awaiting the physical-units upgrade.
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
