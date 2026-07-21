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

from typing import Dict, TYPE_CHECKING

from kl_pipe.priors import (
    Gaussian,
    PriorDict,
    TruncatedNormal,
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
    'F158': {'flux': 120.0, 'rscale': 0.35},
    'F184': {'flux': 100.0, 'rscale': 0.35},
}
# per-band flux prior (mu, sigma, low, high)
_BAND_FLUX_PRIOR = {
    'F087': (100.0, 20.0, 30.0, 250.0),
    'F158': (120.0, 25.0, 30.0, 300.0),
    'F184': (100.0, 20.0, 30.0, 250.0),
}

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


def build_source_model(config: 'ObservationConfig') -> 'SourceModel':
    """SourceModel matching the observation config's bands and lines."""
    from kl_pipe.intensity import InclinedExponentialModel
    from kl_pipe.lines import EmissionLine
    from kl_pipe.source import SourceModel
    from kl_pipe.velocity import CenteredVelocityModel

    for band in config.bands:
        if band not in _BAND_TRUTH:
            raise ValueError(
                f"band '{band}' has no scene defaults; known bands: "
                f"{sorted(_BAND_TRUTH)}"
            )

    return SourceModel(
        velocity_model=CenteredVelocityModel(),
        broadband_models={band: InclinedExponentialModel() for band in config.bands},
        emission_lines={
            'Halpha': EmissionLine(
                intensity=InclinedExponentialModel(),
                continuum=InclinedExponentialModel(),
            )
        },
    )


def scene_truth_defaults(
    config: 'ObservationConfig', fixed_overrides: Dict[str, float]
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
        for par, value in _BAND_TRUTH[band].items():
            truth[f'{band}.{par}'] = value

    # geometry defaults on every component
    for comp in _geometry_components(config):
        truth[f'{comp}.h_over_r'] = 0.1
        truth[f'{comp}.x0'] = 0.0
        truth[f'{comp}.y0'] = 0.0

    # apply spec fixed overrides: broadcast short keys, then dotted keys
    from kl_pipe.ensemble.spec import _BROADCAST_FIXED

    for name, value in fixed_overrides.items():
        if name in _BROADCAST_FIXED:
            for comp in _geometry_components(config):
                truth[f'{comp}.{name}'] = value
        else:
            if name not in truth:
                raise ValueError(
                    f"population.fixed key '{name}' is not a scene parameter; "
                    f"scene parameters: {sorted(truth)}"
                )
            truth[name] = value
    return truth


def scene_priors(
    truth: Dict[str, float], config: 'ObservationConfig', spec: 'EnsembleSpec'
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

    Returns
    -------
    PriorDict
        Sampled priors + fixed values; z pinned to the fit's truth z.
    """
    from kl_pipe.ensemble.spec import DrawSpec

    def population_prior(name: str, draw: DrawSpec):
        if draw.dist == 'uniform':
            return Uniform(draw.params['low'], draw.params['high'])
        if draw.dist == 'lognormal_tf':
            return make_tf_prior(draw.params['center_kms'], draw.params['sigma_tf_dex'])
        raise ValueError(f"no prior rule for draw dist '{draw.dist}' ({name})")

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
        'vel.rscale': TruncatedNormal(truth['vel.rscale'], 0.1, 0.05, 1.0),
        'Halpha.flux': TruncatedNormal(truth['Halpha.flux'], 20.0, 30.0, 250.0),
        'Halpha.rscale': TruncatedNormal(truth['Halpha.rscale'], 0.08, 0.05, 1.0),
        'Halpha.h_over_r': truth['Halpha.h_over_r'],
        'Halpha.x0': TruncatedNormal(0.0, 0.1, -0.5, 0.5),
        'Halpha.y0': TruncatedNormal(0.0, 0.1, -0.5, 0.5),
        'Halpha.dispersion': TruncatedNormal(
            truth['Halpha.dispersion'], 20.0, 5.0, 150.0
        ),
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

    for band in config.bands:
        _, sigma, low, high = _BAND_FLUX_PRIOR[band]
        prior_spec[f'{band}.flux'] = TruncatedNormal(
            truth[f'{band}.flux'], sigma, low, high
        )
        prior_spec[f'{band}.rscale'] = TruncatedNormal(
            truth[f'{band}.rscale'], 0.08, 0.05, 1.0
        )
        prior_spec[f'{band}.h_over_r'] = truth[f'{band}.h_over_r']
        prior_spec[f'{band}.x0'] = TruncatedNormal(0.0, 0.1, -0.5, 0.5)
        prior_spec[f'{band}.y0'] = TruncatedNormal(0.0, 0.1, -0.5, 0.5)

    # cosi: population prior. Stratified axis -> uniform over the stratify
    # range (the bin grid IS the random-orientation population); drawn ->
    # handled by the generic drawn-parameter loop below.
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
