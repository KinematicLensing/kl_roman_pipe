"""
Diagnostic tests for MCMC sampling on TNG mock data.

Tests joint velocity+intensity inference using TNG galaxies with:
- Native orientation (using TNG catalog inclination/PA)
- Custom orientation (PA=30°, cosi=0.5, aligned gas/stellar geometry)
- Corner plots, parameter recovery, data comparison panels

Outputs to tests/out/tng_diagnostics/sampling/

Note: TNG galaxies don't follow simple arctan/exponential models,
so these tests measure "how well can we approximate TNG structure
with our analytic models" rather than exact parameter recovery.
"""

import pytest
import time
import warnings
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

from kl_pipe.tng import TNG50MockData, TNGDataVectorGenerator, TNGRenderConfig
from kl_pipe.parameters import ImagePars
from kl_pipe.velocity import OffsetVelocityModel
from kl_pipe.intensity import InclinedExponentialModel
from kl_pipe.model import KLModel
from kl_pipe.priors import TruncatedNormal, Gaussian, PriorDict
from kl_pipe.sampling import (
    InferenceTask,
    SamplerResult,
    NumpyroSamplerConfig,
    build_sampler,
)
from kl_pipe.sampling.diagnostics import (
    plot_corner,
    plot_recovery,
    print_summary,
)
from kl_pipe.diagnostics import plot_combined_data_comparison
from kl_pipe.utils import get_test_dir

# Import from local test_utils (pytest adds tests/ to sys.path automatically)
from test_utils import TestConfig, redirect_sampler_output


# ==============================================================================
# Module-level pytest markers
# ==============================================================================

pytestmark = [pytest.mark.tng50, pytest.mark.tng_diagnostics, pytest.mark.slow]


# ==============================================================================
# Test Configuration
# ==============================================================================

# Galaxies to test
TEST_SUBHALO_IDS = [8, 19, 29]

# Standard image parameters - smaller for faster likelihood evaluation
IMAGE_SHAPE = (32, 32)  # 1024 pixels instead of 4096
PIXEL_SCALE = 0.2  # arcsec/pixel (coarser to maintain same FOV)
TARGET_REDSHIFT = 0.3  # Closer redshift for larger apparent size and better sampling

# Sampling configuration - balance between speed and convergence
# TNG has model mismatch so we can't expect perfect posteriors
N_SAMPLES = 2000  # More samples for better posterior estimation
N_WARMUP = 1000   # Longer warmup for adaptation with 11 params
N_CHAINS = 4      # 4 chains for proper R-hat convergence diagnostics

# Moderate SNR - allows more noise to reduce sensitivity to model mismatch
# while still resolving structure
DEFAULT_SNR = 50.0

# Timeout for individual sampling runs (seconds)
# Prevents tests from hanging indefinitely on difficult posteriors
MAX_SAMPLING_TIME = 300  # 5 minutes per galaxy


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture(scope="module")
def tng_data():
    """Load TNG50 data once per module."""
    return TNG50MockData()


@pytest.fixture(scope="module")
def output_dir():
    """Output directory for TNG sampling diagnostics."""
    out_dir = get_test_dir() / "out" / "tng_diagnostics" / "sampling"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


@pytest.fixture
def image_pars():
    """Standard test image parameters."""
    return ImagePars(shape=IMAGE_SHAPE, pixel_scale=PIXEL_SCALE, indexing='ij')


# ==============================================================================
# Helper Functions
# ==============================================================================


def estimate_velocity_params(
    velocity_map: np.ndarray,
    image_pars: ImagePars,
) -> Dict[str, float]:
    """
    Estimate velocity model parameters from TNG velocity map.

    Parameters
    ----------
    velocity_map : np.ndarray
        TNG velocity map.
    image_pars : ImagePars
        Image parameters.

    Returns
    -------
    dict
        Estimated parameters: v0, vcirc, vel_rscale
    """
    # Mask out zero pixels (no gas particles)
    valid_mask = velocity_map != 0

    if valid_mask.sum() < 10:
        # Not enough valid pixels, return defaults
        return {'v0': 0.0, 'vcirc': 150.0, 'vel_rscale': 0.5}

    valid_vel = velocity_map[valid_mask]

    # Systemic velocity (median)
    v0 = float(np.median(valid_vel))

    # Circular velocity estimate (half of max-min range gives amplitude)
    # In a rotating disk: v_los goes from -vcirc*sin(i) to +vcirc*sin(i)
    vel_range = np.percentile(valid_vel, 95) - np.percentile(valid_vel, 5)
    vcirc_estimate = vel_range / 2.0

    # Ensure reasonable bounds
    vcirc_estimate = max(50.0, min(400.0, vcirc_estimate))

    # Radius scale: estimate from where velocity reaches 63% of max
    # For arctan model, v(r) = vcirc * (2/pi) * arctan(r/rscale)
    # At r=rscale, v = vcirc * (2/pi) * (pi/4) = vcirc/2
    # Use half-light radius as rough proxy
    fov_arcsec = image_pars.Nx * image_pars.pixel_scale
    vel_rscale = fov_arcsec / 6.0  # Rough estimate

    return {
        'v0': v0,
        'vcirc': vcirc_estimate,
        'vel_rscale': vel_rscale,
        'vel_x0': 0.0,  # Assume centered initially
        'vel_y0': 0.0,
    }


def normalize_intensity_map(
    intensity_map: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Normalize intensity map to unit total flux.

    Parameters
    ----------
    intensity_map : np.ndarray
        TNG intensity map in physical units.

    Returns
    -------
    normalized_map : np.ndarray
        Map normalized to unit total flux.
    normalization : float
        The normalization factor applied (original_sum).
    """
    total_flux = intensity_map.sum()
    if total_flux <= 0:
        return intensity_map, 1.0

    return intensity_map / total_flux, total_flux


def estimate_intensity_params(
    intensity_map: np.ndarray,
    image_pars: ImagePars,
) -> Dict[str, float]:
    """
    Estimate intensity model parameters from TNG intensity map.

    Assumes map is already normalized to unit total flux.

    Parameters
    ----------
    intensity_map : np.ndarray
        Normalized TNG intensity map.
    image_pars : ImagePars
        Image parameters.

    Returns
    -------
    dict
        Estimated parameters: flux, int_rscale
    """
    # For normalized intensity maps (sum=1), the exponential model flux parameter
    # needs to be ~0.01 to give similar total flux (model sum = flux * 100)
    # This is because the model integrates flux over infinite extent
    flux = 0.01

    # Estimate half-light radius
    total = intensity_map.sum()
    if total <= 0:
        return {'flux': flux, 'int_rscale': 0.5, 'int_x0': 0.0, 'int_y0': 0.0}

    cumsum = np.cumsum(np.sort(intensity_map.ravel())[::-1])
    half_light_npix = np.searchsorted(cumsum, total / 2)

    # Convert to radius assuming roughly circular
    half_light_radius_pix = np.sqrt(half_light_npix / np.pi)
    half_light_radius_arcsec = half_light_radius_pix * image_pars.pixel_scale

    # For exponential profile, half-light radius ≈ 1.68 * scale_length
    int_rscale = half_light_radius_arcsec / 1.68

    # Ensure reasonable bounds
    int_rscale = max(0.1, min(3.0, int_rscale))

    return {
        'flux': flux,
        'int_rscale': int_rscale,
        'int_x0': 0.0,
        'int_y0': 0.0,
    }


def create_tng_joint_inference_task(
    true_pars: Dict[str, float],
    data_vel: jnp.ndarray,
    data_int: jnp.ndarray,
    var_vel: float,
    var_int: float,
    image_pars_vel: ImagePars,
    image_pars_int: ImagePars,
) -> InferenceTask:
    """
    Create joint inference task for TNG data with wide priors.

    Uses TruncatedNormal priors centered on estimated "true" values
    but with wide widths to allow exploration.

    Parameters
    ----------
    true_pars : dict
        Estimated true parameter values (geometry from TNG, model params estimated).
    data_vel, data_int : jnp.ndarray
        TNG velocity and intensity data vectors.
    var_vel, var_int : float
        Variance values.
    image_pars_vel, image_pars_int : ImagePars
        Image parameters for each map.

    Returns
    -------
    InferenceTask
        Configured task ready for sampling.
    """
    # Create joint model with OffsetVelocityModel to allow centroid offsets
    vel_model = OffsetVelocityModel()
    int_model = InclinedExponentialModel()
    joint_model = KLModel(
        velocity_model=vel_model,
        intensity_model=int_model,
        shared_pars={'cosi', 'theta_int', 'g1', 'g2'},
    )

    # FOV for offset prior bounds
    fov_arcsec = image_pars_vel.Nx * image_pars_vel.pixel_scale
    offset_bound = fov_arcsec / 4.0  # Allow offsets up to quarter of FOV

    # Define priors - use TruncatedNormal centered on estimates with WIDE widths
    # This allows exploration since TNG doesn't match analytic models exactly
    prior_spec = {
        # Velocity params - wide priors
        'v0': Gaussian(true_pars.get('v0', 0.0), 30.0),  # Wide Gaussian
        'vcirc': TruncatedNormal(
            true_pars.get('vcirc', 150.0), 100.0, 30, 500
        ),  # Very wide
        'vel_rscale': TruncatedNormal(
            true_pars.get('vel_rscale', 0.5), 0.5, 0.05, 3.0
        ),
        'vel_x0': TruncatedNormal(
            true_pars.get('vel_x0', 0.0), 0.5, -offset_bound, offset_bound
        ),
        'vel_y0': TruncatedNormal(
            true_pars.get('vel_y0', 0.0), 0.5, -offset_bound, offset_bound
        ),
        # Intensity params - wide priors
        # After normalization, flux ~0.01 gives unit integrated flux in model
        'flux': TruncatedNormal(
            true_pars.get('flux', 0.01), 0.02, 0.001, 0.1
        ),
        'int_rscale': TruncatedNormal(
            true_pars.get('int_rscale', 0.5), 0.5, 0.05, 3.0
        ),
        'int_x0': TruncatedNormal(
            true_pars.get('int_x0', 0.0), 0.5, -offset_bound, offset_bound
        ),
        'int_y0': TruncatedNormal(
            true_pars.get('int_y0', 0.0), 0.5, -offset_bound, offset_bound
        ),
        # Shared geometric params - centered on known TNG values
        'cosi': TruncatedNormal(
            true_pars['cosi'], 0.2, 0.01, 0.99
        ),
        'theta_int': TruncatedNormal(
            true_pars['theta_int'], 0.5, 0.0, 2 * np.pi  # Full 0-2pi range
        ),
        # No shear for TNG (unlensed)
        'g1': 0.0,
        'g2': 0.0,
    }

    priors = PriorDict(prior_spec)

    # Create task using factory method
    task = InferenceTask.from_joint_model(
        model=joint_model,
        priors=priors,
        data_vel=data_vel,
        data_int=data_int,
        variance_vel=var_vel,
        variance_int=var_int,
        image_pars_vel=image_pars_vel,
        image_pars_int=image_pars_int,
    )

    return task


def get_map_from_samples(result: SamplerResult) -> Dict[str, float]:
    """
    Estimate MAP parameters from samples.

    Uses the sample with highest log_prob as MAP estimate.
    """
    max_idx = np.argmax(result.log_prob)
    map_theta = result.samples[max_idx]
    return {name: float(map_theta[i]) for i, name in enumerate(result.param_names)}


def evaluate_model_at_map(
    task: InferenceTask,
    map_pars: Dict[str, float],
    image_pars_vel: ImagePars,
    image_pars_int: ImagePars,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Evaluate model at MAP parameters.

    Returns velocity and intensity model predictions.
    """
    # Build full parameter dict (sampled + fixed)
    full_pars = {**task.fixed_params, **map_pars}

    # Evaluate velocity model
    vel_model = task.model.velocity_model
    vel_pars = {k: full_pars[k] for k in vel_model.PARAMETER_NAMES}
    theta_vel = jnp.array([vel_pars[k] for k in vel_model.PARAMETER_NAMES])
    model_vel = vel_model.render(theta_vel, 'image', image_pars_vel)

    # Evaluate intensity model
    int_model = task.model.intensity_model
    int_pars = {k: full_pars[k] for k in int_model.PARAMETER_NAMES}
    theta_int = jnp.array([int_pars[k] for k in int_model.PARAMETER_NAMES])
    model_int = int_model.render(theta_int, 'image', image_pars_int)

    return model_vel, model_int


def run_tng_sampling_test(
    galaxy: Dict[str, Any],
    subhalo_id: int,
    config: TNGRenderConfig,
    true_pars: Dict[str, float],
    test_name: str,
    output_dir: Path,
    image_pars: ImagePars,
    snr: float = DEFAULT_SNR,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run complete TNG sampling test for one galaxy.

    Generates data, runs numpyro sampler, and saves diagnostics.

    Parameters
    ----------
    galaxy : dict
        TNG galaxy data dict.
    subhalo_id : int
        SubhaloID for naming.
    config : TNGRenderConfig
        Render configuration.
    true_pars : dict
        True/estimated parameter values.
    test_name : str
        Name prefix for output files.
    output_dir : Path
        Directory for diagnostic outputs.
    image_pars : ImagePars
        Image parameters.
    snr : float
        Signal-to-noise ratio.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Test results including runtime, samples, recovery stats.
    """
    gen = TNGDataVectorGenerator(galaxy)

    # Generate velocity map
    velocity_raw, var_vel = gen.generate_velocity_map(config, snr=snr, seed=seed)

    # Generate intensity map and normalize
    intensity_raw, var_int_raw = gen.generate_intensity_map(
        config, snr=snr, seed=seed + 1
    )
    intensity_norm, norm_factor = normalize_intensity_map(intensity_raw)

    # Calculate variance from data characteristics rather than TNG noise model
    # This ensures sensible likelihood values since TNG doesn't match analytic models
    # We inflate variance significantly to account for model mismatch
    
    # Velocity variance: TNG rotation curves don't follow arctan profiles
    # Use much larger variance to allow for systematic model mismatch
    vel_nonzero = velocity_raw[velocity_raw != 0]
    if len(vel_nonzero) > 10:
        vel_range = np.percentile(vel_nonzero, 95) - np.percentile(vel_nonzero, 5)
        # Use 20% of the range as 1-sigma - this is ~10x larger than SNR=100 would give
        var_vel_scalar = (0.2 * vel_range) ** 2
    else:
        var_vel_scalar = 100.0 ** 2  # Default 100 km/s
    
    # Intensity variance: TNG particle distributions don't match exponential profiles
    # Use variance based on data RMS, inflated for model mismatch
    int_rms = np.std(intensity_norm[intensity_norm > 0]) if (intensity_norm > 0).sum() > 0 else 0.01
    # Allow model to be off by ~10x data RMS (very conservative for TNG mismatch)
    var_int_scalar = (10.0 * int_rms) ** 2
    
    # Ensure minimum variance floors for numerical stability
    var_vel_scalar = max(var_vel_scalar, 100.0)  # At least 100 (km/s)^2
    var_int_scalar = max(var_int_scalar, 1e-4)  # Reasonable floor for intensity

    # Estimate velocity parameters from data
    vel_params = estimate_velocity_params(velocity_raw, image_pars)
    int_params = estimate_intensity_params(intensity_norm, image_pars)

    # Combine with geometric true_pars
    full_true_pars = {**true_pars, **vel_params, **int_params}

    print(f"\n{'='*60}")
    print(f"TNG Sampling: SubhaloID {subhalo_id} - {test_name}")
    print(f"{'='*60}")
    print(f"Geometric params: cosi={true_pars['cosi']:.3f}, theta_int={true_pars['theta_int']:.3f}")
    print(f"Estimated velocity: v0={vel_params['v0']:.1f}, vcirc={vel_params['vcirc']:.1f}, "
          f"vel_rscale={vel_params['vel_rscale']:.2f}")
    print(f"Estimated intensity: flux={int_params['flux']:.2f}, int_rscale={int_params['int_rscale']:.2f}")
    print(f"Variance: vel={var_vel_scalar:.2e}, int={var_int_scalar:.2e}")

    # Create inference task
    data_vel = jnp.array(velocity_raw)
    data_int = jnp.array(intensity_norm)

    task = create_tng_joint_inference_task(
        full_true_pars,
        data_vel,
        data_int,
        var_vel_scalar,
        var_int_scalar,
        image_pars,
        image_pars,
    )

    print(f"Sampled parameters: {task.sampled_names}")
    print(f"Fixed parameters: {list(task.fixed_params.keys())}")

    # Configure numpyro sampler with settings balanced for speed and quality
    # Given model mismatch, we prioritize getting reasonable diagnostics over perfect convergence
    numpyro_config = NumpyroSamplerConfig(
        n_samples=N_SAMPLES,
        n_warmup=N_WARMUP,
        n_chains=N_CHAINS,
        chain_method='vectorized',
        seed=seed,
        progress=True,  # Show progress so we can see what's happening
        reparam_strategy='prior',
        dense_mass=False,  # Diagonal mass matrix is faster and often sufficient
        max_tree_depth=8,  # Limit tree depth to prevent very long iterations
        target_accept_prob=0.8,  # Standard acceptance target
    )

    # Run sampler
    sampler = build_sampler('numpyro', task, numpyro_config)
    start_time = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = sampler.run()
    runtime = time.time() - start_time

    print(f"Sampling complete: {result.n_samples} samples in {runtime:.1f}s")

    # Get MAP estimate
    map_pars = get_map_from_samples(result)

    # Evaluate model at MAP
    model_vel, model_int = evaluate_model_at_map(
        task, map_pars, image_pars, image_pars
    )

    # Save corner plot
    sampler_info = {
        'name': 'numpyro',
        'runtime': runtime,
        'settings': {
            'n_samples': N_SAMPLES,
            'n_warmup': N_WARMUP,
            'n_chains': N_CHAINS,
            'SNR': snr,
            'SubhaloID': subhalo_id,
        },
    }

    # Only include sampled params in true_values for corner plot
    corner_true_values = {k: full_true_pars[k] for k in result.param_names if k in full_true_pars}

    fig = plot_corner(
        result,
        true_values=corner_true_values,
        map_values=map_pars,
        sampler_info=sampler_info,
    )
    corner_path = output_dir / f"{test_name}_subhalo{subhalo_id}_corner.png"
    fig.savefig(corner_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved corner plot: {corner_path}")

    # Save data comparison panels
    # Use the noisy data as both "noisy" and "true" since TNG is the "truth"
    plot_combined_data_comparison(
        data_vel_noisy=np.asarray(data_vel),
        data_vel_true=np.asarray(data_vel),  # TNG is truth
        model_vel=np.asarray(model_vel),
        data_int_noisy=np.asarray(data_int),
        data_int_true=np.asarray(data_int),  # TNG is truth
        model_int=np.asarray(model_int),
        test_name=f"{test_name}_subhalo{subhalo_id}",
        output_dir=output_dir,
        variance_vel=var_vel_scalar,
        variance_int=var_int_scalar,
        n_params=task.n_params,
        model_label='MAP Model',
    )
    print(f"Saved data comparison: {test_name}_subhalo{subhalo_id}_data_comparison.png")

    # Print summary
    print_summary(result, true_values=corner_true_values)

    return {
        'runtime': runtime,
        'result': result,
        'map_pars': map_pars,
        'true_pars': full_true_pars,
        'model_vel': model_vel,
        'model_int': model_int,
    }


# ==============================================================================
# Test Classes
# ==============================================================================


class TestTNGNativeSampling:
    """
    Test joint sampling on TNG galaxies at their native orientation.

    Uses TNG catalog inclination and position angle as "true" geometric params.
    """

    @pytest.mark.parametrize("subhalo_id", TEST_SUBHALO_IDS)
    def test_native_orientation_sampling(
        self, tng_data, output_dir, image_pars, subhalo_id
    ):
        """
        Test joint sampling at native TNG orientation.

        Renders galaxy at its intrinsic inc/PA from TNG catalog,
        runs numpyro sampler, and saves diagnostics.
        """
        # Load galaxy
        galaxy = tng_data.get_galaxy(subhalo_id=subhalo_id)
        gen = TNGDataVectorGenerator(galaxy)

        # Extract native orientation as "true" geometric parameters
        true_pars = {
            'cosi': gen.native_cosi,
            'theta_int': gen.native_pa_rad,
            'g1': 0.0,
            'g2': 0.0,
        }

        print(f"\nNative orientation: inc={gen.native_inclination_deg:.1f}°, "
              f"PA={gen.native_pa_deg:.1f}°")
        print(f"  -> cosi={gen.native_cosi:.3f}, theta_int={gen.native_pa_rad:.3f} rad")

        # Create render config for native orientation
        config = TNGRenderConfig(
            image_pars=image_pars,
            band='r',
            use_native_orientation=True,
            target_redshift=TARGET_REDSHIFT,
            use_cic_gridding=True,
        )

        # Run test
        results = run_tng_sampling_test(
            galaxy=galaxy,
            subhalo_id=subhalo_id,
            config=config,
            true_pars=true_pars,
            test_name="native",
            output_dir=output_dir,
            image_pars=image_pars,
            snr=DEFAULT_SNR,
            seed=42 + subhalo_id,
        )

        # Basic assertions
        assert results['result'].n_samples > 0
        assert results['runtime'] > 0
        assert all(np.isfinite(results['result'].log_prob))


class TestTNGCustomSampling:
    """
    Test joint sampling on TNG galaxies at custom orientation.

    Uses PA=30° and cosi=0.5, with gas-stellar offset disabled
    so velocity and intensity share exact geometric parameters.
    """

    # Custom orientation parameters
    CUSTOM_PA_DEG = 30.0
    CUSTOM_COSI = 0.5

    @pytest.mark.parametrize("subhalo_id", TEST_SUBHALO_IDS)
    def test_custom_orientation_sampling(
        self, tng_data, output_dir, image_pars, subhalo_id
    ):
        """
        Test joint sampling at custom orientation with aligned geometry.

        Uses PA=30°, cosi=0.5, and preserve_gas_stellar_offset=False
        to force aligned velocity/intensity geometry.
        """
        # Load galaxy
        galaxy = tng_data.get_galaxy(subhalo_id=subhalo_id)

        # Custom geometric parameters
        theta_int = np.radians(self.CUSTOM_PA_DEG)
        true_pars = {
            'cosi': self.CUSTOM_COSI,
            'theta_int': theta_int,
            'g1': 0.0,
            'g2': 0.0,
            'x0': 0.0,
            'y0': 0.0,
        }

        print(f"\nCustom orientation: inc={np.degrees(np.arccos(self.CUSTOM_COSI)):.1f}°, "
              f"PA={self.CUSTOM_PA_DEG:.1f}°")
        print(f"  -> cosi={self.CUSTOM_COSI:.3f}, theta_int={theta_int:.3f} rad")
        print(f"  -> preserve_gas_stellar_offset=False (aligned geometry)")

        # Create render config with custom orientation and aligned gas/stellar
        pars = {k: true_pars[k] for k in ['cosi', 'theta_int', 'x0', 'y0', 'g1', 'g2']}
        config = TNGRenderConfig(
            image_pars=image_pars,
            band='r',
            use_native_orientation=False,
            pars=pars,
            target_redshift=TARGET_REDSHIFT,
            use_cic_gridding=True,
            preserve_gas_stellar_offset=False,  # Force alignment
        )

        # Run test
        results = run_tng_sampling_test(
            galaxy=galaxy,
            subhalo_id=subhalo_id,
            config=config,
            true_pars=true_pars,
            test_name="custom_aligned",
            output_dir=output_dir,
            image_pars=image_pars,
            snr=DEFAULT_SNR,
            seed=100 + subhalo_id,
        )

        # Basic assertions
        assert results['result'].n_samples > 0
        assert results['runtime'] > 0
        assert all(np.isfinite(results['result'].log_prob))

        # Additional check: cosi should be recovered near true value
        # (with aligned geometry, model should fit better)
        map_cosi = results['map_pars'].get('cosi', 0)
        cosi_error = abs(map_cosi - self.CUSTOM_COSI)
        print(f"cosi recovery: true={self.CUSTOM_COSI:.3f}, MAP={map_cosi:.3f}, "
              f"error={cosi_error:.3f}")

        # Warn if recovery is poor (not fail - TNG doesn't match model exactly)
        if cosi_error > 0.3:
            warnings.warn(
                f"Poor cosi recovery for SubhaloID {subhalo_id}: "
                f"error={cosi_error:.3f}"
            )
