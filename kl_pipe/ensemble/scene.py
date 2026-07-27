"""
Canonical galaxy scene for ensemble fits.

One place defines (a) the full dotted truth-parameter defaults for a scene
assembled from an ObservationConfig's bands and lines, (b) the fit prior rules,
and (c) the SourceModel structure. The expander overrides the per-fit varying
truths (stratified cosi, drawn theta_int / vel.vcirc / z, injected g1 / g2)
on top of these defaults; the worker builds the matching fit priors.

Values mirror the flagship joint config (tests/test_flagship.py and
experiments/sweverett/vista_kit/tasks_vista.py) so the infra shakedown fits
the same scene the benchmarks measured. Observation-level production values
are a separate, pending decision.

Prior policy: drawn truth parameters get their generating distribution as the
fit prior (self-consistent by construction); nuisance parameters that do not
vary across the population keep the flagship priors centered on the scene
defaults;
z is pinned to the per-fit truth in v1.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, TYPE_CHECKING

from kl_pipe.ensemble.population import (
    BULGE_CLASSICAL_N,
    BULGE_PSEUDO_N,
    BULGE_PSEUDO_WEIGHT,
    BULGE_SIZE_RATIO_LN_SCATTER,
    BULGE_SIZE_RATIO_MEDIAN,
    HALPHA_RSCALE_RATIO_DEX,
    HALPHA_RSCALE_RATIO_MEDIAN,
    VEL_RSCALE_RATIO_DEX,
    VEL_RSCALE_RATIO_MEDIAN,
)
from kl_pipe.priors import (
    ConditionalLogNormal,
    Gaussian,
    LogNormal,
    PriorDict,
    TruncatedLogNormal,
    TruncatedNormal,
    TruncatedNormalMixture,
    Uniform,
    make_tf_prior,
)

if TYPE_CHECKING:
    from kl_pipe.ensemble.spec import EnsembleSpec, ObservationConfig
    from kl_pipe.source import SourceModel

# per-band scene defaults (flagship values; F184 mirrors F087, slightly
# fainter than F158)
_BAND_TRUTH = {
    'F087': {'flux': 100.0, 'rscale': 0.3},
    'F106': {'flux': 105.0, 'rscale': 0.31},
    'F129': {'flux': 110.0, 'rscale': 0.33},
    'F158': {'flux': 120.0, 'rscale': 0.35},
    'F184': {'flux': 100.0, 'rscale': 0.35},
}
# per-band flux prior (mu, sigma, low, high)
_BAND_FLUX_PRIOR = {
    'F087': (100.0, 20.0, 30.0, 250.0),
    'F106': (105.0, 21.0, 30.0, 265.0),
    'F129': (110.0, 22.0, 30.0, 275.0),
    'F158': (120.0, 25.0, 30.0, 300.0),
    'F184': (100.0, 20.0, 30.0, 250.0),
}

# Vertical structure, pinned on both sides (painted and fit at the same
# value), so it is a stated model assumption rather than a recovered
# quantity. The disk value is the GalSim InclinedExponential default.
# Hoffmann et al. 2022 measure C/A = 0.24 for disc-dominated galaxies below
# z = 1, which maps to h_over_r 0.42-0.53 through the renderer; adopting that
# would cost about 14% of the apparent-shape lever overall and 38% in the
# lowest-inclination bin, so it is carried as an open systematic.
_DISK_H_OVER_R = 0.1
# The bulge has no anchor of its own. Costantin et al. 2018 measure C/A = 0.65
# for CALIFA bulges; the conversion to a scale-height ratio is approximate and
# a bulge capped at B/T = 0.3 carries little of the inclination signal.
_BULGE_H_OVER_HLR = 0.5
# base bulge truth defaults; the catalog expander overrides bulge_frac and
# bulge_hlr per galaxy from the catalog columns
_BULGE_FRAC_DEFAULT = 0.1
_BULGE_HLR_DEFAULT = 0.2  # arcsec
# overridden per galaxy by the catalog expander from the bulge paint
_BULGE_N_SERSIC_DEFAULT = 4.0

# shared scene defaults (flagship values)
_SHARED_TRUTH = {
    'vel.v0': 10.0,
    'vel.rscale': 0.3,
    'Halpha.flux': 100.0,
    'Halpha.rscale': 0.25,
    'Halpha.dispersion': 50.0,
    'Halpha.cont.flux_per_nm': 25.0,
    'Halpha.cont.rscale': 0.25,
}


# components carrying the broadcastable geometry params (h_over_r, x0, y0)
def _geometry_components(config: 'ObservationConfig') -> tuple:
    return tuple(config.bands) + ('Halpha', 'Halpha.cont')


def build_source_model(
    config: 'ObservationConfig',
    bulge_nsersic: Optional[float] = None,
    sample_bulge_nsersic: bool = False,
) -> 'SourceModel':
    """SourceModel matching the observation config's bands and lines.

    Parameters
    ----------
    config : ObservationConfig
        Defines the scene's bands and lines.
    bulge_nsersic : float, optional
        If set (catalog mode), broadband bands carry a bulge: each band is a
        ``BulgeDiskModel`` with shared disk/bulge centroid and shear applied
        to both components. The Halpha line and its continuum stay disk-only
        (the line traces the star-forming disk, not the old bulge). If None
        (sampled mode), every component is a single
        ``InclinedExponentialModel`` as before.
    sample_bulge_nsersic : bool
        If True, the bulge Sersic index becomes the sampled parameter
        ``{band}.bulge_n_sersic`` rather than being fixed at the painted
        value. ``bulge_nsersic`` then only decides whether the scene has a
        bulge and supplies the truth; it is not handed to the model, since
        pinning a fit parameter at its own truth flatters the recovery.
        Requires ``bulge_nsersic`` to be set.
    """
    from kl_pipe.intensity import BulgeDiskModel, InclinedExponentialModel
    from kl_pipe.lines import EmissionLine
    from kl_pipe.source import SourceModel
    from kl_pipe.velocity import CenteredVelocityModel

    for band in config.bands:
        if band not in _BAND_TRUTH:
            raise ValueError(
                f"band '{band}' has no scene defaults; known bands: "
                f"{sorted(_BAND_TRUTH)}"
            )

    if bulge_nsersic is None:
        if sample_bulge_nsersic:
            raise ValueError(
                "sample_bulge_nsersic requires bulge_nsersic to be set: it "
                "decides whether the scene has a bulge at all"
            )
        broadband_models = {band: InclinedExponentialModel() for band in config.bands}
    else:
        broadband_models = {
            band: BulgeDiskModel(
                bulge_nsersic=None if sample_bulge_nsersic else bulge_nsersic,
                shared_centroids=True,
                shear_bulge=True,
            )
            for band in config.bands
        }

    return SourceModel(
        velocity_model=CenteredVelocityModel(),
        broadband_models=broadband_models,
        emission_lines={
            'Halpha': EmissionLine(
                intensity=InclinedExponentialModel(),
                continuum=InclinedExponentialModel(),
            )
        },
    )


def scene_truth_defaults(
    config: 'ObservationConfig',
    fixed_overrides: Dict[str, float],
    bulge_bands: bool = False,
) -> Dict[str, float]:
    """
    Full dotted truth defaults for the scene, before per-fit overrides.

    Parameters
    ----------
    config : ObservationConfig
        Defines which bands (and lines) the scene contains.
    fixed_overrides : dict
        The spec's resolved population.fixed block. Short keys 'h_over_r', 'x0',
        'y0' broadcast to every scene component; dotted keys override a
        single parameter. Unknown dotted keys raise.
    bulge_bands : bool
        If True (catalog mode), broadband bands are BulgeDiskModel and carry
        the composite truth keys (total_flux, bulge_frac, disk_rscale,
        disk_h_over_r, bulge_hlr, bulge_h_over_hlr, x0, y0). bulge_frac and
        bulge_hlr get placeholder defaults here; the catalog expander
        overrides them per galaxy. If False (sampled mode), bands are
        single-disk (flux, rscale, h_over_r, x0, y0).

    Returns
    -------
    dict
        Dotted truth parameters, excluding the per-fit varying set
        (cosi, theta_int, g1, g2, vel.vcirc, z) which the expander fills.
    """
    truth: Dict[str, float] = dict(_SHARED_TRUTH)
    for band in config.bands:
        if band not in _BAND_TRUTH:
            raise ValueError(
                f"band '{band}' has no scene defaults; known bands: "
                f"{sorted(_BAND_TRUTH)}"
            )
        bt = _BAND_TRUTH[band]
        if bulge_bands:
            truth[f'{band}.total_flux'] = bt['flux']
            truth[f'{band}.bulge_frac'] = _BULGE_FRAC_DEFAULT
            truth[f'{band}.disk_rscale'] = bt['rscale']
            truth[f'{band}.disk_h_over_r'] = _DISK_H_OVER_R
            truth[f'{band}.bulge_hlr'] = _BULGE_HLR_DEFAULT
            truth[f'{band}.bulge_h_over_hlr'] = _BULGE_H_OVER_HLR
            truth[f'{band}.bulge_n_sersic'] = _BULGE_N_SERSIC_DEFAULT
            truth[f'{band}.x0'] = 0.0
            truth[f'{band}.y0'] = 0.0
        else:
            truth[f'{band}.flux'] = bt['flux']
            truth[f'{band}.rscale'] = bt['rscale']
            truth[f'{band}.h_over_r'] = _DISK_H_OVER_R
            truth[f'{band}.x0'] = 0.0
            truth[f'{band}.y0'] = 0.0

    # geometry defaults on the line components (bands handled above, since
    # their thickness key differs between single-disk and bulge-disk)
    for comp in ('Halpha', 'Halpha.cont'):
        truth[f'{comp}.h_over_r'] = _DISK_H_OVER_R
        truth[f'{comp}.x0'] = 0.0
        truth[f'{comp}.y0'] = 0.0

    # apply spec fixed overrides: broadcast short keys, then dotted keys.
    # broadcast only updates keys the scene actually has (bulge bands carry
    # disk_h_over_r, not h_over_r), so a broadcast never invents orphan keys
    from kl_pipe.ensemble.spec import _BROADCAST_FIXED

    for name, value in fixed_overrides.items():
        if name in _BROADCAST_FIXED:
            for comp in _geometry_components(config):
                key = f'{comp}.{name}'
                if key in truth:
                    truth[key] = value
        else:
            if name not in truth:
                raise ValueError(
                    f"population.fixed key '{name}' is not a scene parameter; "
                    f"scene parameters: {sorted(truth)}"
                )
            truth[name] = value
    return truth


# catalog-mode rscale prior bounds [arcsec]. The sampled-mode (0.05, 1.0)
# bounds do NOT contain the catalog truths: the dev population's selected
# disk scale lengths span 0.0086-1.18 arcsec (flagship2_dev, snr_line >= 5
# selection, measured 2026-07-21), so bounds are set below/above the
# observed extremes rather than silently clipping truth out of support
_CATALOG_RSCALE_LOW = 0.005
_CATALOG_RSCALE_HIGH = 2.0

# Population distributions replacing the truth-centered priors, fitted to the
# selected Flagship2 sample (flagship2_dev at snr_line_total_min 20, n = 400,
# measured 2026-07-25). log10 median and log10 scatter.
_CATALOG_RSCALE_LOG10_MU = -0.672
_CATALOG_RSCALE_LOG10_SIGMA = 0.237
_CONT_FLUX_LOG10_MU = 0.770
_CONT_FLUX_LOG10_SIGMA = 0.305
# support wide enough to contain the selected sample's continuum amplitudes,
# which span roughly 1.3-93 in the scene's internal flux units per nm
_CONT_FLUX_LOW = 0.05
_CONT_FLUX_HIGH = 400.0
# bulge half-light radius support. Matched to the disk floor rather than set
# lower: the smallest allowed size drives the worst-case maxk, so an
# unnecessarily small floor enlarges the render grid on every evaluation.
# The population's bulge sizes reach about 0.006 arcsec at four sigma.
_BULGE_HLR_LOW = 0.005
_BULGE_HLR_HIGH = 2.0

# Systemic velocity prior width. One grism pixel at 1.1 nm is about 200 km/s
# at the observed Halpha wavelength; the line centroid is measured to roughly
# 14 km/s at line SNR 20 and 40 km/s at SNR 7, so this leaves v0 data-
# dominated across the sample rather than encoding knowledge we do not have.
_V0_PRIOR_SIGMA_KMS = 200.0

# informative population priors for the bulge decomposition. A flat
# bulge_frac (Uniform) + flat bulge_hlr (LogUniform) add no curvature along
# the bulge amplitude/size directions, which the data leaves near-degenerate
# whenever the bulge is faint or unresolved. That flatness both floors the
# NUTS mass matrix and carves spurious MAP optima -> catastrophic divergences.
# Informative population priors inject curvature exactly where the likelihood
# is flat, without centering on truth: honest marginalization (assert the
# population, not the answer), regularizing without biasing shear.
#
# bulge_frac: matched to the selected population. Flagship2 dev, z 0.55-1.9,
# two-component disks, B/T <= 0.3 (the bulge_fraction_max selection cut):
# B/T mean 0.080, std 0.080, n=206k (measured 2026-07-23; stable against a
# brightest-decile flux cut). The truncation follows the spec's own
# bulge_fraction_max selection cut (prior support == selected population
# support); loc/scale stay fixed at the B/T <= 0.3 measurement, which the
# production specs use.
_BULGE_FRAC_LOC = 0.08
_BULGE_FRAC_SCALE = 0.08
# bulge_hlr: exactly the generating distribution of the population's bulge
# size paint (population.py BULGE_SIZE_RATIO_*): a LogNormal on the bulge-to-
# disk size ratio, median set per galaxy from the disk size. Prior ==
# generating distribution by construction, so the marginalization is exact.
# The paint caps the ratio below 1 while the prior is uncapped; the cap
# removes only the >3-sigma tail (ln(1/0.3)/0.4 = 3.0), a negligible
# mismatch that keeps the prior JIT-simple and the render grid finite.
_EXP_R50_OVER_RSCALE = 1.6783  # exponential-disk r50 / scale-length ratio

# bulge_n_sersic: prior == the distribution the paint draws from, so the
# marginalization is exact, as for bulge_hlr above.


def _bulge_nsersic_prior() -> TruncatedNormalMixture:
    """Bulge Sersic index prior: the population paint's own mixture."""
    return TruncatedNormalMixture(
        (
            TruncatedNormal(*BULGE_PSEUDO_N),
            TruncatedNormal(*BULGE_CLASSICAL_N),
        ),
        (BULGE_PSEUDO_WEIGHT, 1.0 - BULGE_PSEUDO_WEIGHT),
    )


def scene_priors(
    truth: Dict[str, float],
    config: 'ObservationConfig',
    spec: 'EnsembleSpec',
    row: Optional[Dict] = None,
) -> PriorDict:
    """
    Fit priors for one fit, self-consistent with the generating population.

    Parameters
    ----------
    truth : dict
        The fit's fully-resolved dotted truth (from the manifest row).
    config : ObservationConfig
        Scene structure (bands).
    spec : EnsembleSpec
        Supplies the population distributions for stratified/drawn params.
    row : dict or pd.Series, optional
        The manifest row. Required for catalog populations, whose vcirc
        prior is observable-conditioned (``pop.prior_vcirc_*`` columns);
        ignored for sampled populations.

    Returns
    -------
    PriorDict
        Sampled priors + fixed values; z pinned to the fit's truth z.
    """
    from kl_pipe.ensemble.spec import DrawSpec

    is_catalog = spec.catalog_population is not None
    if is_catalog and row is None:
        raise ValueError(
            "catalog-mode priors require the manifest row (pop.prior_vcirc_* "
            "columns); pass row"
        )

    def population_prior(name: str, draw: DrawSpec):
        if draw.dist == 'uniform':
            return Uniform(draw.params['low'], draw.params['high'])
        if draw.dist == 'lognormal_tf':
            return make_tf_prior(draw.params['center_kms'], draw.params['sigma_tf_dex'])
        raise ValueError(f"no prior rule for draw dist '{draw.dist}' ({name})")

    # catalog truths carry the catalog disk scale length, which exceeds the
    # sampled-mode rscale bounds (see _CATALOG_RSCALE_* provenance)
    rscale_low = _CATALOG_RSCALE_LOW if is_catalog else 0.05
    rscale_high = _CATALOG_RSCALE_HIGH if is_catalog else 1.0

    prior_spec: Dict[str, object] = {
        # injected shear: wide, uninformative prior so posterior widths
        # reflect the data's shear constraint, not the prior; unbounded --
        # truncation would re-inject a prior edge (matches the flagship).
        # Width is spec-configurable (wider -> more data-driven sigma_eps).
        'g1': Gaussian(0.0, spec.shear_fit_prior_sigma),
        'g2': Gaussian(0.0, spec.shear_fit_prior_sigma),
        # nuisance kinematics (flagship prior widths/bounds, centered on the
        # fit's truth -- identical to scene defaults unless the spec fixed
        # block overrides them)
        'vel.v0': Gaussian(truth['vel.v0'], 10.0),
        'vel.rscale': TruncatedNormal(
            truth['vel.rscale'], 0.1, rscale_low, rscale_high
        ),
        'Halpha.flux': TruncatedNormal(truth['Halpha.flux'], 20.0, 30.0, 250.0),
        'Halpha.rscale': TruncatedNormal(
            truth['Halpha.rscale'], 0.08, rscale_low, rscale_high
        ),
        'Halpha.h_over_r': truth['Halpha.h_over_r'],
        'Halpha.x0': TruncatedNormal(0.0, 0.1, -0.5, 0.5),
        'Halpha.y0': TruncatedNormal(0.0, 0.1, -0.5, 0.5),
        'Halpha.dispersion': TruncatedNormal(
            truth['Halpha.dispersion'], 20.0, 5.0, 150.0
        ),
        # catalog truths (line flux / EW_obs) span 1.3-93 flux/nm on the dev
        # population (flagship2_dev, snr_line >= 5, measured 2026-07-21), so
        # the (0, 200) support contains every truth
        'Halpha.cont.flux_per_nm': TruncatedNormal(
            truth['Halpha.cont.flux_per_nm'], 15.0, 0.0, 200.0
        ),
        # continuum shape pinned to truth (flagship convention)
        'Halpha.cont.rscale': truth['Halpha.cont.rscale'],
        'Halpha.cont.h_over_r': truth['Halpha.cont.h_over_r'],
        'Halpha.cont.x0': truth['Halpha.cont.x0'],
        'Halpha.cont.y0': truth['Halpha.cont.y0'],
        # z pinned to the per-fit truth (v1; sampled-narrow-z is planned)
        'z': truth['z'],
    }

    # catalog broadband is BulgeDiskModel only when the bulge paint is on; a
    # disk-only twin (paint.bulge: false) takes the single-disk branch with
    # the catalog rscale bounds
    bulge_bands = is_catalog and spec.catalog_population.paint_bulge

    for band in config.bands:
        _, sigma, low, high = _BAND_FLUX_PRIOR[band]
        prior_spec[f'{band}.x0'] = TruncatedNormal(0.0, 0.1, -0.5, 0.5)
        prior_spec[f'{band}.y0'] = TruncatedNormal(0.0, 0.1, -0.5, 0.5)
        if bulge_bands:
            # catalog broadband = BulgeDiskModel. Sampled: total_flux,
            # disk_rscale (truth-centered, interim convention), bulge_frac,
            # bulge_hlr. bulge_frac + bulge_hlr use informative POPULATION
            # priors (not truth-centered); disk_h_over_r + bulge_h_over_hlr fixed.
            prior_spec[f'{band}.total_flux'] = TruncatedNormal(
                truth[f'{band}.total_flux'], sigma, low, high
            )
            prior_spec[f'{band}.disk_rscale'] = TruncatedNormal(
                truth[f'{band}.disk_rscale'], 0.08, rscale_low, rscale_high
            )
            prior_spec[f'{band}.disk_h_over_r'] = truth[f'{band}.disk_h_over_r']
            bf_max = spec.catalog_population.bulge_fraction_max
            prior_spec[f'{band}.bulge_frac'] = TruncatedNormal(
                _BULGE_FRAC_LOC,
                _BULGE_FRAC_SCALE,
                0.0,
                1.0 if bf_max is None else float(bf_max),
            )
            disk_r50 = _EXP_R50_OVER_RSCALE * truth[f'{band}.disk_rscale']
            bulge_hlr_median = BULGE_SIZE_RATIO_MEDIAN * disk_r50
            prior_spec[f'{band}.bulge_hlr'] = LogNormal(
                math.log(bulge_hlr_median), BULGE_SIZE_RATIO_LN_SCATTER
            )
            prior_spec[f'{band}.bulge_h_over_hlr'] = truth[f'{band}.bulge_h_over_hlr']
            if spec.sample_bulge_nsersic:
                prior_spec[f'{band}.bulge_n_sersic'] = _bulge_nsersic_prior()
        else:
            prior_spec[f'{band}.flux'] = TruncatedNormal(
                truth[f'{band}.flux'], sigma, low, high
            )
            prior_spec[f'{band}.rscale'] = TruncatedNormal(
                truth[f'{band}.rscale'], 0.08, rscale_low, rscale_high
            )
            prior_spec[f'{band}.h_over_r'] = truth[f'{band}.h_over_r']

    if is_catalog:
        cp = spec.catalog_population
        # observable-conditioned TFR prior: mu = TFR evaluated at the NOISY
        # simulated photometric mass (logm_obs), so it is mis-centered from
        # the fit's truth vcirc by construction; width = intrinsic TFR
        # scatter + propagated mass error (in dex, combined upstream)
        prior_spec['vel.vcirc'] = make_tf_prior(
            float(row['pop.prior_vcirc_mu_kms']),
            float(row['pop.prior_vcirc_sigma_dex']),
        )
        # orientation: the generating distributions (isotropic redraw)
        prior_spec['cosi'] = Uniform(*cp.cosi_range)
        prior_spec['theta_int'] = Uniform(0.0, math.pi)
        # self-consistent population prior on the painted dispersion:
        # sigma0(z) = intercept + slope*z with the paint scatter
        # (Ubler+2019 affine evolution); bounds = the paint floor and the
        # shared 150 km/s scene ceiling
        prior_spec['Halpha.dispersion'] = TruncatedNormal(
            cp.sigma0_intercept_kms + cp.sigma0_slope_kms * truth['z'],
            cp.sigma0_scatter_kms,
            cp.sigma0_min_kms,
            150.0,
        )

        # Sizes. The catalog disk scale is the only measured size, so the
        # broadband disk carries the population distribution and the line and
        # rotation-curve scales condition on the sampled value through the
        # same ratio distributions the paint used. Conditioning on a sampled
        # parameter marginalizes with the rest, so no measurement is reused.
        ln10 = math.log(10.0)
        parent = f'{config.bands[0]}.disk_rscale'
        for band in config.bands:
            key = f'{band}.disk_rscale' if bulge_bands else f'{band}.rscale'
            prior_spec[key] = TruncatedLogNormal(
                _CATALOG_RSCALE_LOG10_MU * ln10,
                _CATALOG_RSCALE_LOG10_SIGMA * ln10,
                rscale_low,
                rscale_high,
            )
        if not bulge_bands:
            parent = f'{config.bands[0]}.rscale'
        prior_spec['vel.rscale'] = ConditionalLogNormal(
            parent,
            math.log(VEL_RSCALE_RATIO_MEDIAN),
            VEL_RSCALE_RATIO_DEX * ln10,
            rscale_low,
            rscale_high,
        )
        prior_spec['Halpha.rscale'] = ConditionalLogNormal(
            parent,
            math.log(HALPHA_RSCALE_RATIO_MEDIAN),
            HALPHA_RSCALE_RATIO_DEX * ln10,
            rscale_low,
            rscale_high,
        )
        if bulge_bands:
            # bulge size relative to its own band's disk, from the same paint
            # constants, so the truth-derived median is gone
            for band in config.bands:
                prior_spec[f'{band}.bulge_hlr'] = ConditionalLogNormal(
                    f'{band}.disk_rscale',
                    math.log(BULGE_SIZE_RATIO_MEDIAN * _EXP_R50_OVER_RSCALE),
                    BULGE_SIZE_RATIO_LN_SCATTER,
                    _BULGE_HLR_LOW,
                    _BULGE_HLR_HIGH,
                )

        # continuum amplitude: population distribution of the catalog
        # equivalent widths, which is both leak-free and tighter than the
        # truth-centered prior it replaces
        prior_spec['Halpha.cont.flux_per_nm'] = TruncatedLogNormal(
            _CONT_FLUX_LOG10_MU * ln10,
            _CONT_FLUX_LOG10_SIGMA * ln10,
            _CONT_FLUX_LOW,
            _CONT_FLUX_HIGH,
        )

        # systemic velocity: deliberately wider than the painted offset. The
        # grism measures the line centroid to tens of km/s, so a prior about
        # one grism pixel wide leaves this direction data-dominated.
        prior_spec['vel.v0'] = Gaussian(0.0, _V0_PRIOR_SIGMA_KMS)
    else:
        # cosi: population prior. Stratified axis -> uniform over the
        # stratify range (the bin grid IS the random-orientation
        # population); drawn -> handled by the drawn-parameter loop below.
        if spec.stratify_param == 'cosi':
            prior_spec['cosi'] = Uniform(*spec.stratify_range)

        # drawn params: generating distribution = fit prior (self-consistent)
        for name, draw in spec.draw.items():
            if name == 'z':
                continue  # z is pinned above in v1
            prior_spec[name] = population_prior(name, draw)

    if 'cosi' not in prior_spec:
        raise ValueError(
            "cosi has no prior: it must be either the stratified axis or a "
            "population.draw entry"
        )
    if 'theta_int' not in prior_spec:
        raise ValueError(
            "spec population.draw must include theta_int (position angle population)"
        )
    if 'vel.vcirc' not in prior_spec:
        raise ValueError(
            "spec population.draw must include vcirc (Tully-Fisher population)"
        )

    return PriorDict(prior_spec)


# per-fit varying truth parameters the expander fills (everything else comes
# from scene_truth_defaults)
VARYING_TRUTH_PARAMS = ('cosi', 'theta_int', 'g1', 'g2', 'vel.vcirc', 'z')
