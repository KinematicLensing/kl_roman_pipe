"""
Tests for central JAX host-device-count control (kl_pipe._devices,
KLPIPE_CPU_DEVICES).

Default: importing kl_pipe leaves JAX device discovery untouched.
KLPIPE_CPU_DEVICES=N forces N host CPU devices, enabling
chain_method='parallel' multi-chain MCMC on CPU. Subprocess isolation is
used wherever the env var must influence the one-time import-side
configuration, or where actual device count matters.
"""

import os
import subprocess
import sys

import pytest

# repo root: subprocesses run here so ``import kl_pipe`` resolves to THIS
# checkout (cwd precedes any editable-install path for ``python -c``)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_python(code: str, extra_env: dict) -> subprocess.CompletedProcess:
    """Run ``code`` in a fresh interpreter with ``extra_env`` overlaid."""
    env = dict(os.environ)
    env.update(extra_env)
    return subprocess.run(
        [sys.executable, '-c', code],
        env=env,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )


class TestDefaultBehavior:
    def test_unset_env_is_noop(self):
        """Unset KLPIPE_CPU_DEVICES: device count is JAX's own default."""
        result = _run_python(
            "import jax; n_before = len(jax.devices()); "
            "import kl_pipe; "
            "print(len(jax.devices()) == n_before)",
            extra_env={'KLPIPE_CPU_DEVICES': ''},
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == 'True'

    def test_default_in_process(self):
        """In-process default: kl_pipe import doesn't touch device count."""
        import jax

        import kl_pipe  # noqa: F401

        # cannot assert a specific count here (depends on other tests /
        # process history), just that importing kl_pipe doesn't raise.
        assert len(jax.devices()) >= 1


class TestEnvOptIn:
    def test_forces_requested_device_count(self):
        """KLPIPE_CPU_DEVICES=4 yields exactly 4 host CPU devices."""
        result = _run_python(
            "import kl_pipe; import jax; print(len(jax.devices()))",
            extra_env={'KLPIPE_CPU_DEVICES': '4'},
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == '4'

    def test_single_device_request(self):
        """KLPIPE_CPU_DEVICES=1 is a valid (if trivial) request."""
        result = _run_python(
            "import kl_pipe; import jax; print(len(jax.devices()))",
            extra_env={'KLPIPE_CPU_DEVICES': '1'},
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == '1'


class TestInvalidValue:
    @pytest.mark.parametrize('value', ['0', '-1', 'banana', '2.5'])
    def test_invalid_env_value_raises_in_helper(self, monkeypatch, value):
        """cpu_devices_requested() raises ValueError on unrecognized values."""
        from kl_pipe._devices import cpu_devices_requested

        monkeypatch.setenv('KLPIPE_CPU_DEVICES', value)
        with pytest.raises(ValueError, match='KLPIPE_CPU_DEVICES'):
            cpu_devices_requested()

    @pytest.mark.parametrize('value', ['0', '-1', 'banana', '2.5'])
    def test_invalid_env_value_fails_import(self, value):
        """Package import fails loudly on an unrecognized KLPIPE_CPU_DEVICES."""
        result = _run_python(
            "import kl_pipe",
            extra_env={'KLPIPE_CPU_DEVICES': value},
        )
        assert result.returncode != 0
        assert 'ValueError' in result.stderr
        assert 'KLPIPE_CPU_DEVICES' in result.stderr


class TestSetTooLate:
    def test_set_after_eager_op_raises_runtime_error(self):
        """Setting jax_num_cpu_devices after backend init raises loudly
        (not caught/softened by kl_pipe)."""
        result = _run_python(
            "import jax; import jax.numpy as jnp; jnp.array(1.0) + 1; "
            "import kl_pipe",
            extra_env={'KLPIPE_CPU_DEVICES': '4'},
        )
        assert result.returncode != 0
        assert 'RuntimeError' in result.stderr


class TestIdempotence:
    def test_configure_cpu_devices_repeat_call_is_noop(self):
        """Repeat configure_cpu_devices() calls never re-raise or reconfigure."""
        from kl_pipe._devices import configure_cpu_devices

        configure_cpu_devices()
        configure_cpu_devices()  # must not raise


class TestParallelChainEndToEnd:
    def test_klpipe_cpu_devices_resolves_to_parallel(self):
        """KLPIPE_CPU_DEVICES=2 + chain_method=None on a 2-chain velocity-only
        task auto-resolves to 'parallel' and produces finite samples.

        Kept small (12x12 grid, 20 warmup/20 samples per chain) to run in
        well under 60s.
        """
        code = """
import os
import numpy as np
import jax
import jax.numpy as jnp

import kl_pipe
from kl_pipe.velocity import CenteredVelocityModel
from kl_pipe.parameters import ImagePars
from kl_pipe.synthetic import SyntheticVelocity
from kl_pipe.priors import Gaussian, TruncatedNormal, PriorDict
from kl_pipe.source import SourceModel
from kl_pipe.observation import build_velocity_obs
from kl_pipe.sampling import InferenceTask, NumpyroSamplerConfig, build_sampler

image_pars = ImagePars(shape=(12, 12), pixel_scale=0.4, indexing='ij')
true_pars_flat = {
    'v0': 10.0, 'vcirc': 200.0, 'rscale': 5.0, 'cosi': 0.6,
    'theta_int': 0.785, 'g1': 0.02, 'g2': -0.01,
}
synth_vel = SyntheticVelocity(true_pars_flat, model_type='arctan', seed=42)
data_vel = synth_vel.generate(image_pars, snr=1000)
var_vel = synth_vel.variance

source = SourceModel(velocity_model=CenteredVelocityModel())
priors = PriorDict({
    'vel.v0': Gaussian(10.0, 5.0),
    'vel.vcirc': TruncatedNormal(200.0, 50.0, 100, 300),
    'vel.rscale': TruncatedNormal(5.0, 2.0, 0.4, 20.0),
    'cosi': TruncatedNormal(0.6, 0.2, 0.01, 0.99),
    'theta_int': TruncatedNormal(0.785, 0.3, 0, np.pi),
    'g1': 0.02,
    'g2': -0.01,
})
vel_obs = build_velocity_obs(
    image_pars, data=jnp.array(data_vel), variance=var_vel
)
task = InferenceTask.from_obs(source, priors, velocity_obs=vel_obs)

config = NumpyroSamplerConfig(
    n_samples=20, n_warmup=20, n_chains=2, chain_method=None,
    seed=42, progress=False,
)
sampler = build_sampler('numpyro', task, config)
result = sampler.run()

print('n_devices', jax.local_device_count())
print('chain_method', result.metadata['chain_method'])
print('n_samples', result.n_samples)
print('all_finite', bool(np.all(np.isfinite(result.samples))))
"""
        result = _run_python(code, extra_env={'KLPIPE_CPU_DEVICES': '2'})
        assert result.returncode == 0, result.stderr
        lines = dict(line.split(' ', 1) for line in result.stdout.strip().splitlines())
        assert lines['n_devices'] == '2'
        assert lines['chain_method'] == 'parallel'
        assert lines['n_samples'] == '40'  # 20 samples * 2 chains
        assert lines['all_finite'] == 'True'
