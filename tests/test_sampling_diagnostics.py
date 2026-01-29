"""Diagnostic tests for MCMC sampling infrastructure.

Tests sampling on joint velocity+intensity models with visual diagnostics:
- Corner plots with true value markers
- Data comparison panels (noisy/true/MAP model)
- Parameter comparison plots
- Sampler comparison (emcee vs nautilus vs numpyro)

All tests produce diagnostic outputs in tests/out/sampling/.
"""

import pytest
import time
import warnings
import numpy as np
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, Optional

from kl_pipe.velocity import CenteredVelocityModel
from kl_pipe.intensity import InclinedExponentialModel
from kl_pipe.model import KLModel
from kl_pipe.parameters import ImagePars
from kl_pipe.synthetic import SyntheticVelocity, SyntheticIntensity
from kl_pipe.priors import Uniform, Gaussian, TruncatedNormal, PriorDict
from kl_pipe.sampling import (
    InferenceTask,
    SamplerResult,
    EnsembleSamplerConfig,
    NestedSamplerConfig,
    NumpyroSamplerConfig,
    build_sampler,
)
from kl_pipe.sampling.diagnostics import (
    plot_corner,
    plot_trace,
    plot_recovery,
    plot_corner_comparison,
    print_summary,
)
from kl_pipe.diagnostics import (
    plot_combined_data_comparison,
    plot_data_comparison_panels,
    compute_joint_nsigma,
    nsigma_to_color,
)
from kl_pipe.utils import get_test_dir

# Import from local test_utils (pytest adds tests/ to sys.path automatically)
from test_utils import (
    TestConfig,
    redirect_sampler_output,
)


# ==============================================================================
# Test Configuration
# ==============================================================================

# Baseline sampler for comparison tests - gets C0 (blue), others follow in order
BASELINE_SAMPLER = 'numpyro'

# Skip nautilus by default (set INCLUDE_NAUTILUS=1 to include)
import os

INCLUDE_NAUTILUS = os.environ.get('INCLUDE_NAUTILUS', '0') == '1'


@pytest.fixture(scope="module")
def test_config():
    """Test configuration fixture for sampling diagnostics."""
    out_dir = get_test_dir() / "out" / "sampling"
    config = TestConfig(out_dir, include_poisson_noise=False)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    return config


# ==============================================================================
# Helper Functions
# ==============================================================================


def get_map_from_samples(result: SamplerResult) -> Dict[str, float]:
    """
    Estimate MAP parameters from samples.

    Uses the sample with highest log_prob as MAP estimate.

    Parameters
    ----------
    result : SamplerResult
        Sampling result.

    Returns
    -------
    dict
        MAP estimate for each sampled parameter.
    """
    max_idx = np.argmax(result.log_prob)
    map_theta = result.samples[max_idx]
    return {name: float(map_theta[i]) for i, name in enumerate(result.param_names)}


def get_median_from_samples(result: SamplerResult) -> Dict[str, float]:
    """
    Get posterior median for each parameter.

    Parameters
    ----------
    result : SamplerResult
        Sampling result.

    Returns
    -------
    dict
        Posterior median for each sampled parameter.
    """
    summary = result.get_summary()
    return {name: summary[name]['quantiles'][0.5] for name in result.param_names}


def generate_joint_synthetic_data(
    true_pars: Dict[str, float],
    image_pars_vel: ImagePars,
    image_pars_int: ImagePars,
    snr: float,
    seed: int = 42,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, float, float]:
    """
    Generate joint velocity + intensity synthetic data.

    Parameters
    ----------
    true_pars : dict
        True parameter values for both models.
    image_pars_vel : ImagePars
        Image parameters for velocity map.
    image_pars_int : ImagePars
        Image parameters for intensity map.
    snr : float
        Signal-to-noise ratio for both maps.
    seed : int
        Random seed.

    Returns
    -------
    data_vel_true : jnp.ndarray
        True velocity map.
    data_vel_noisy : jnp.ndarray
        Noisy velocity map.
    data_int_true : jnp.ndarray
        True intensity map.
    data_int_noisy : jnp.ndarray
        Noisy intensity map.
    var_vel : float
        Velocity variance.
    var_int : float
        Intensity variance.
    """
    # Extract velocity parameters
    vel_model = CenteredVelocityModel()
    vel_pars = {k: v for k, v in true_pars.items() if k in vel_model.PARAMETER_NAMES}

    synth_vel = SyntheticVelocity(vel_pars, model_type='arctan', seed=seed)
    data_vel_noisy = synth_vel.generate(
        image_pars_vel, snr=snr, seed=seed, include_poisson=False
    )
    data_vel_true = synth_vel.data_true
    var_vel = synth_vel.variance

    # Extract intensity parameters
    int_model = InclinedExponentialModel()
    int_pars = {k: v for k, v in true_pars.items() if k in int_model.PARAMETER_NAMES}

    synth_int = SyntheticIntensity(int_pars, model_type='exponential', seed=seed + 1)
    data_int_noisy = synth_int.generate(
        image_pars_int, snr=snr, seed=seed + 1, include_poisson=False
    )
    data_int_true = synth_int.data_true
    var_int = synth_int.variance

    return (
        jnp.array(data_vel_true),
        jnp.array(data_vel_noisy),
        jnp.array(data_int_true),
        jnp.array(data_int_noisy),
        var_vel,
        var_int,
    )


def create_joint_inference_task(
    true_pars: Dict[str, float],
    data_vel: jnp.ndarray,
    data_int: jnp.ndarray,
    var_vel: float,
    var_int: float,
    image_pars_vel: ImagePars,
    image_pars_int: ImagePars,
    sample_shear: bool = True,
) -> InferenceTask:
    """
    Create joint inference task with appropriate priors.

    Parameters
    ----------
    true_pars : dict
        True parameter values.
    data_vel, data_int : jnp.ndarray
        Observed data.
    var_vel, var_int : float
        Variances.
    image_pars_vel, image_pars_int : ImagePars
        Image parameters.
    sample_shear : bool
        Whether to sample shear parameters (g1, g2).

    Returns
    -------
    InferenceTask
        Configured task ready for sampling.
    """
    # Create joint model
    vel_model = CenteredVelocityModel()
    int_model = InclinedExponentialModel()
    joint_model = KLModel(
        velocity_model=vel_model,
        intensity_model=int_model,
        shared_pars={'cosi', 'theta_int', 'g1', 'g2'},
    )

    # Define priors - use TruncatedNormal for bounded parameters to ensure
    # gradient-based samplers (BlackJAX) have well-defined gradients.
    # Uniform priors have zero gradient in the interior, which causes NUTS to stall.
    prior_spec = {
        # Velocity params
        'v0': Gaussian(true_pars['v0'], 5.0),
        'vcirc': TruncatedNormal(200.0, 50.0, 100, 300),
        'vel_rscale': TruncatedNormal(5.0, 2.0, 1.0, 10.0),
        # Intensity params
        'flux': TruncatedNormal(1.0, 1.0, 0.1, 5.0),
        'int_rscale': TruncatedNormal(3.0, 2.0, 0.5, 10.0),
        'int_x0': 0.0,  # Fixed
        'int_y0': 0.0,  # Fixed
        # Shared geometric params
        'cosi': TruncatedNormal(0.5, 0.3, 0.01, 0.99),
        'theta_int': TruncatedNormal(np.pi / 2, 1.0, 0, np.pi),
    }

    # Shear: sample or fix
    if sample_shear:
        prior_spec['g1'] = TruncatedNormal(0.0, 0.05, -0.1, 0.1)
        prior_spec['g2'] = TruncatedNormal(0.0, 0.05, -0.1, 0.1)
    else:
        prior_spec['g1'] = true_pars['g1']
        prior_spec['g2'] = true_pars['g2']

    priors = PriorDict(prior_spec)

    # Create task
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


def evaluate_model_at_map(
    task: InferenceTask,
    map_pars: Dict[str, float],
    image_pars_vel: ImagePars,
    image_pars_int: ImagePars,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Evaluate model at MAP parameters.

    Parameters
    ----------
    task : InferenceTask
        Inference task.
    map_pars : dict
        MAP parameter values.
    image_pars_vel, image_pars_int : ImagePars
        Image parameters.

    Returns
    -------
    model_vel, model_int : jnp.ndarray
        Model evaluations at MAP.
    """
    # Build full parameter dict (sampled + fixed)
    full_pars = {**task.fixed_params, **map_pars}

    # Evaluate velocity model using render interface
    vel_model = task.model.velocity_model
    vel_pars = {k: full_pars[k] for k in vel_model.PARAMETER_NAMES}
    theta_vel = jnp.array([vel_pars[k] for k in vel_model.PARAMETER_NAMES])
    model_vel = vel_model.render(theta_vel, 'image', image_pars_vel)

    # Evaluate intensity model using render interface
    int_model = task.model.intensity_model
    int_pars = {k: full_pars[k] for k in int_model.PARAMETER_NAMES}
    theta_int = jnp.array([int_pars[k] for k in int_model.PARAMETER_NAMES])
    model_int = int_model.render(theta_int, 'image', image_pars_int)

    return model_vel, model_int


def save_summary_table(
    results: Dict[str, SamplerResult],
    timings: Dict[str, float],
    true_values: Dict[str, float],
    output_path: Path,
) -> None:
    """
    Save summary statistics table as text file.

    Parameters
    ----------
    results : dict
        Mapping of sampler_name -> SamplerResult.
    timings : dict
        Mapping of sampler_name -> time in seconds.
    true_values : dict
        True parameter values.
    output_path : Path
        Output file path.
    """
    lines = []
    lines.append("=" * 90)
    lines.append("SAMPLER COMPARISON SUMMARY")
    lines.append("=" * 90)

    for sampler_name, result in results.items():
        lines.append(f"\n{sampler_name.upper()}")
        lines.append(
            f"Time: {timings[sampler_name]:.1f}s | N samples: {result.n_samples}"
        )
        if result.evidence is not None:
            lines.append(f"Log evidence: {result.evidence:.2f}")
        if result.acceptance_fraction is not None:
            lines.append(f"Acceptance: {result.acceptance_fraction:.1%}")
        lines.append("-" * 70)

        summary = result.get_summary()
        lines.append(
            f"{'Parameter':<12} {'True':>10} {'Mean':>10} {'Std':>10} "
            f"{'16%':>10} {'84%':>10} {'Error':>8}"
        )
        lines.append("-" * 70)

        for name in result.param_names:
            s = summary[name]
            true_val = true_values.get(name, float('nan'))
            if abs(true_val) > 1e-10:
                rel_error = abs(s['mean'] - true_val) / abs(true_val)
                error_str = f"{rel_error:.1%}"
            else:
                error_str = "N/A"

            lines.append(
                f"{name:<12} {true_val:>10.4f} {s['mean']:>10.4f} {s['std']:>10.4f} "
                f"{s['quantiles'][0.16]:>10.4f} {s['quantiles'][0.84]:>10.4f} {error_str:>8}"
            )

    lines.append("\n" + "=" * 90)

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))


# ==============================================================================
# Joint Sampling Diagnostic Tests
# ==============================================================================


@pytest.mark.slow
class TestJointSamplingDiagnostics:
    """
    Diagnostic tests for joint velocity+intensity sampling.

    Produces corner plots, data comparison panels, and parameter recovery
    diagnostics for each SNR value.
    """

    # Image parameters
    IMAGE_PARS_VEL = ImagePars(shape=(24, 24), pixel_scale=0.4, indexing='ij')
    IMAGE_PARS_INT = ImagePars(shape=(32, 32), pixel_scale=0.3, indexing='ij')

    # Sampler config - need enough iterations for convergence
    SAMPLER_CONFIG = EnsembleSamplerConfig(
        n_walkers=32,
        n_iterations=2000,
        burn_in=500,
        thin=1,
        seed=42,
        progress=False,
    )

    # Looser config for shear tests (more degeneracy)
    SAMPLER_CONFIG_SHEAR = EnsembleSamplerConfig(
        n_walkers=48,
        n_iterations=3000,
        burn_in=1000,
        thin=1,
        seed=42,
        progress=False,
    )

    @pytest.mark.parametrize("snr", [100, 50, 10])
    def test_joint_sampling_no_shear(self, snr, test_config):
        """Test joint sampling without shear (g1=g2=0)."""
        true_pars = {
            'v0': 10.0,
            'vcirc': 200.0,
            'vel_rscale': 5.0,
            'cosi': 0.6,
            'theta_int': 0.785,
            'g1': 0.0,
            'g2': 0.0,
            'flux': 1.0,
            'int_rscale': 3.0,
            'int_x0': 0.0,
            'int_y0': 0.0,
        }
        self._run_joint_sampling_test(
            true_pars, snr, test_config, "joint_no_shear", sample_shear=False
        )

    @pytest.mark.parametrize("snr", [100, 50, 10])
    def test_joint_sampling_with_shear(self, snr, test_config):
        """Test joint sampling with shear (g1=0.03, g2=-0.02)."""
        true_pars = {
            'v0': 10.0,
            'vcirc': 200.0,
            'vel_rscale': 5.0,
            'cosi': 0.6,
            'theta_int': 0.785,
            'g1': 0.03,
            'g2': -0.02,
            'flux': 1.0,
            'int_rscale': 3.0,
            'int_x0': 0.0,
            'int_y0': 0.0,
        }
        self._run_joint_sampling_test(
            true_pars, snr, test_config, "joint_with_shear", sample_shear=True
        )

    def _run_joint_sampling_test(
        self,
        true_pars: Dict[str, float],
        snr: float,
        config: TestConfig,
        scenario: str,
        sample_shear: bool,
    ):
        """
        Core implementation for joint sampling diagnostic tests.

        Parameters
        ----------
        true_pars : dict
            True parameter values.
        snr : float
            Signal-to-noise ratio.
        config : TestConfig
            Test configuration.
        scenario : str
            Scenario name for output files.
        sample_shear : bool
            Whether to sample shear parameters.
        """
        test_name = f"{scenario}_snr{snr}"
        test_dir = config.output_dir / test_name
        test_dir.mkdir(parents=True, exist_ok=True)

        # Generate synthetic data
        (
            data_vel_true,
            data_vel_noisy,
            data_int_true,
            data_int_noisy,
            var_vel,
            var_int,
        ) = generate_joint_synthetic_data(
            true_pars,
            self.IMAGE_PARS_VEL,
            self.IMAGE_PARS_INT,
            snr=snr,
            seed=42,
        )

        # Create inference task
        task = create_joint_inference_task(
            true_pars,
            data_vel_noisy,
            data_int_noisy,
            var_vel,
            var_int,
            self.IMAGE_PARS_VEL,
            self.IMAGE_PARS_INT,
            sample_shear=sample_shear,
        )

        # Run sampler - use more iterations when sampling shear
        sampler_config = (
            self.SAMPLER_CONFIG_SHEAR if sample_shear else self.SAMPLER_CONFIG
        )
        sampler = build_sampler('emcee', task, sampler_config)
        start_time = time.time()
        result = sampler.run()
        runtime = time.time() - start_time

        # Get MAP estimate
        map_pars = get_map_from_samples(result)

        # Evaluate model at MAP
        model_vel, model_int = evaluate_model_at_map(
            task, map_pars, self.IMAGE_PARS_VEL, self.IMAGE_PARS_INT
        )

        # =====================================================================
        # Generate Diagnostics
        # =====================================================================

        # 1. Corner plot with sampler info, true values (black), and MAP (red)
        sampler_info = {
            'name': 'emcee',
            'runtime': runtime,
            'settings': {
                'n_walkers': sampler_config.n_walkers,
                'n_iterations': sampler_config.n_iterations,
                'SNR': snr,
            },
        }
        fig = plot_corner(
            result,
            true_values=true_pars,
            map_values=map_pars,
            sampler_info=sampler_info,
        )
        fig.savefig(test_dir / f"{test_name}_corner.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

        # 2. Combined data comparison panels (velocity + intensity stacked)
        plot_combined_data_comparison(
            data_vel_noisy=np.asarray(data_vel_noisy),
            data_vel_true=np.asarray(data_vel_true),
            model_vel=np.asarray(model_vel),
            data_int_noisy=np.asarray(data_int_noisy),
            data_int_true=np.asarray(data_int_true),
            model_int=np.asarray(model_int),
            test_name=test_name,
            output_dir=test_dir,
            variance_vel=var_vel,
            variance_int=var_int,
            n_params=task.n_params,
            model_label='MAP Model',
        )

        # 3. Parameter recovery plot with joint Nσ validation
        fig, recovery_stats = plot_recovery(
            result,
            true_pars,
            output_path=test_dir / f"{test_name}_parameter_recovery.png",
            sampler_name='emcee',
        )
        plt.close(fig)

        # Print summary and joint Nσ
        print(f"\n{'='*60}")
        print(f"Test: {test_name}")
        print(
            f"Joint Nσ: {recovery_stats['joint_nsigma']:.2f} "
            f"(color: {recovery_stats['title_color']})"
        )
        print(f"{'='*60}")
        print_summary(result, true_values=true_pars)

        # Validate using joint Nσ instead of per-parameter tolerances
        # 2σ = warning, 3σ = fail
        joint_nsigma = recovery_stats['joint_nsigma']
        if joint_nsigma > 3.0:
            pytest.fail(
                f"Joint Nσ = {joint_nsigma:.2f} > 3σ threshold. "
                f"Recovery failed for {test_name}."
            )
        elif joint_nsigma > 2.0:
            warnings.warn(
                f"Joint Nσ = {joint_nsigma:.2f} > 2σ threshold for {test_name}. "
                f"May indicate suboptimal convergence."
            )


# ==============================================================================
# Sampler Comparison Test
# ==============================================================================


@pytest.mark.slow
class TestSamplerComparison:
    """Compare all three main samplers on the same joint problem.

    Samplers compared:
    - emcee: Affine-invariant ensemble MCMC (robust, moderate speed)
    - nautilus: Nested sampling (provides evidence, good for multimodal)
    - numpyro: NUTS with Z-score reparameterization (fast, gradient-based)

    Produces overlaid corner plot and summary statistics table.
    """

    # Image parameters
    IMAGE_PARS_VEL = ImagePars(shape=(24, 24), pixel_scale=0.4, indexing='ij')
    IMAGE_PARS_INT = ImagePars(shape=(32, 32), pixel_scale=0.3, indexing='ij')

    @pytest.mark.parametrize('snr', [50, 1])
    def test_sampler_comparison(self, test_config, snr):
        """Run all samplers and compare results at different SNR values."""
        test_name = f"sampler_comparison_snr{snr}"
        test_dir = test_config.output_dir / test_name
        test_dir.mkdir(parents=True, exist_ok=True)

        # Fixed problem: joint with shear
        true_pars = {
            'v0': 10.0,
            'vcirc': 200.0,
            'vel_rscale': 5.0,
            'cosi': 0.6,
            'theta_int': 0.785,
            'g1': 0.03,
            'g2': -0.02,
            'flux': 1.0,
            'int_rscale': 3.0,
            'int_x0': 0.0,
            'int_y0': 0.0,
        }

        # Generate synthetic data
        (
            data_vel_true,
            data_vel_noisy,
            data_int_true,
            data_int_noisy,
            var_vel,
            var_int,
        ) = generate_joint_synthetic_data(
            true_pars,
            self.IMAGE_PARS_VEL,
            self.IMAGE_PARS_INT,
            snr=snr,
            seed=42,
        )

        # Create inference task
        task = create_joint_inference_task(
            true_pars,
            data_vel_noisy,
            data_int_noisy,
            var_vel,
            var_int,
            self.IMAGE_PARS_VEL,
            self.IMAGE_PARS_INT,
            sample_shear=True,
        )

        results = {}
        timings = {}

        # =====================================================================
        # Run emcee
        # =====================================================================
        print("\nRunning emcee...")
        start = time.time()
        config_emcee = EnsembleSamplerConfig(
            n_walkers=48,
            n_iterations=2500,
            burn_in=500,
            seed=42,
            progress=False,
        )
        sampler = build_sampler('emcee', task, config_emcee)
        emcee_log = test_config.get_sampler_log_path(test_name, 'emcee')
        with redirect_sampler_output(
            emcee_log, also_terminal=test_config.verbose_terminal
        ):
            results['emcee'] = sampler.run()
        timings['emcee'] = time.time() - start
        print(f"emcee completed in {timings['emcee']:.1f}s")

        # =====================================================================
        # Run nautilus (tuned: n_live=500 for ~8 params, n_networks=4)
        # Skipped by default - set INCLUDE_NAUTILUS=1 to run
        # =====================================================================
        if INCLUDE_NAUTILUS:
            print("\nRunning nautilus...")
            start = time.time()
            config_nautilus = NestedSamplerConfig(
                n_live=500,  # ~50x n_params for reliable convergence
                n_networks=4,  # Default network ensemble size
                seed=42,
                progress=False,
                verbose=False,
            )
            sampler = build_sampler('nautilus', task, config_nautilus)
            nautilus_log = test_config.get_sampler_log_path(test_name, 'nautilus')
            with redirect_sampler_output(
                nautilus_log, also_terminal=test_config.verbose_terminal
            ):
                results['nautilus'] = sampler.run()
            timings['nautilus'] = time.time() - start
            print(f"nautilus completed in {timings['nautilus']:.1f}s")
            if results['nautilus'].evidence is not None:
                print(f"Log evidence: {results['nautilus'].evidence:.2f}")
        else:
            print("\nSkipping nautilus (set INCLUDE_NAUTILUS=1 to run)")

        # =====================================================================
        # Run NumPyro (vectorized chains for speed)
        # To manually adjust samples/warmup for tighter constraints, edit lines below
        # =====================================================================
        print("\nRunning numpyro...")
        start = time.time()
        config_numpyro = NumpyroSamplerConfig(
            n_samples=1000,  # Increase for tighter constraints
            n_warmup=500,  # Increase for better adaptation
            n_chains=4,
            chain_method='vectorized',  # Run chains in parallel on single device
            seed=42,
            progress=False,
            reparam_strategy='prior',
            dense_mass=True,
        )
        sampler = build_sampler('numpyro', task, config_numpyro)
        numpyro_log = test_config.get_sampler_log_path(test_name, 'numpyro')
        with redirect_sampler_output(
            numpyro_log, also_terminal=test_config.verbose_terminal
        ):
            results['numpyro'] = sampler.run()
        timings['numpyro'] = time.time() - start
        print(f"numpyro completed in {timings['numpyro']:.1f}s")

        # Report NumPyro diagnostics
        if results['numpyro'].diagnostics:
            diag = results['numpyro'].diagnostics
            n_div = diag.get('n_divergences', 0)
            step_size = diag.get('step_size', None)
            print(f"Divergences: {n_div}")
            if step_size is not None:
                print(f"Step size: {step_size:.4f}")

        # =====================================================================
        # Generate Comparison Diagnostics (all three samplers)
        # =====================================================================

        # Sampler configs for display in corner plot
        sampler_configs = {
            'emcee': {'n_walkers': 48, 'n_iter': 2500},
            'nautilus': {'n_live': 500, 'n_networks': 4},
            'numpyro': {'n_chains': 2, 'n_warmup': 200},
        }

        # 1. Overlaid corner plot with timings and sampler configs
        fig = plot_corner_comparison(
            results,
            true_values=true_pars,
            timings=timings,
            baseline_sampler=BASELINE_SAMPLER,
            sampler_configs=sampler_configs,
            output_path=test_dir / f"{test_name}_corner.png",
        )
        plt.close(fig)

        # 2. Per-sampler data vector plots
        for sampler_name, result in results.items():
            map_pars = get_map_from_samples(result)
            model_vel, model_int = evaluate_model_at_map(
                task, map_pars, self.IMAGE_PARS_VEL, self.IMAGE_PARS_INT
            )
            plot_combined_data_comparison(
                data_vel_noisy=np.asarray(data_vel_noisy),
                data_vel_true=np.asarray(data_vel_true),
                model_vel=np.asarray(model_vel),
                data_int_noisy=np.asarray(data_int_noisy),
                data_int_true=np.asarray(data_int_true),
                model_int=np.asarray(model_int),
                test_name=f"{test_name}_{sampler_name}",
                output_dir=test_dir,
                variance_vel=var_vel,
                variance_int=var_int,
                n_params=task.n_params,
                model_label=f'{sampler_name} MAP',
            )

        # 3. Per-sampler parameter recovery plots with joint Nσ
        all_recovery_stats = {}
        for sampler_name, result in results.items():
            fig, recovery_stats = plot_recovery(
                result,
                true_pars,
                output_path=test_dir / f"{test_name}_{sampler_name}_recovery.png",
                sampler_name=sampler_name,
            )
            plt.close(fig)
            all_recovery_stats[sampler_name] = recovery_stats
            print(f"\n{sampler_name}: Joint Nσ = {recovery_stats['joint_nsigma']:.2f}")

        # 4. Summary table
        save_summary_table(
            results,
            timings,
            true_pars,
            test_dir / f"{test_name}_summary.txt",
        )

        # 5. Individual summaries
        for name, result in results.items():
            print(f"\n{'='*60}")
            print(f"Sampler: {name}")
            print(f"Time: {timings[name]:.1f}s")
            print(f"Joint Nσ: {all_recovery_stats[name]['joint_nsigma']:.2f}")
            print(f"{'='*60}")
            print_summary(result, true_values=true_pars)

        # Validate using joint Nσ:
        # - baseline sampler: 2σ = warning, 3σ = fail (strict, as it's our reference)
        # - other samplers: 2σ = warning, 3σ = loud warning (don't fail, just alert)
        for name, stats in all_recovery_stats.items():
            nsigma = stats['joint_nsigma']
            is_baseline = name == BASELINE_SAMPLER

            if nsigma > 3.0:
                if is_baseline:
                    pytest.fail(
                        f"{name} (BASELINE): Joint Nσ = {nsigma:.2f} > 3σ threshold. "
                        f"Recovery failed - baseline sampler must recover parameters."
                    )
                else:
                    warnings.warn(
                        f"⚠️ {name}: Joint Nσ = {nsigma:.2f} > 3σ threshold! "
                        f"Non-baseline sampler showing significant deviation. "
                        f"Check sampler configuration or increase samples.",
                        UserWarning,
                    )
            elif nsigma > 2.0:
                warnings.warn(
                    f"{name}: Joint Nσ = {nsigma:.2f} > 2σ threshold. "
                    f"May indicate suboptimal convergence."
                )
