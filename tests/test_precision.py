"""
Tests for central precision control (kl_pipe._precision, KLPIPE_FP32).

Default: importing kl_pipe (or any submodule) enables JAX x64 so arrays
default to float64. KLPIPE_FP32=1 opts into JAX's native float32 default.
Subprocess isolation is used wherever the env var must influence the
one-time import-side configuration.
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


class TestDefaultPrecision:
    def test_default_import_yields_float64(self):
        """In-process default: kl_pipe import enables x64, arrays are float64."""
        import jax
        import jax.numpy as jnp

        import kl_pipe  # noqa: F401

        assert jax.config.jax_enable_x64
        assert jnp.asarray(1.0).dtype == jnp.float64

    def test_submodule_import_configures_precision(self):
        """Direct submodule import (fresh interpreter) still enables x64."""
        result = _run_python(
            "import kl_pipe.coordinates; import jax.numpy as jnp; "
            "print(jnp.asarray(1.0).dtype)",
            extra_env={'KLPIPE_FP32': ''},
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == 'float64'


class TestFP32OptIn:
    @pytest.mark.parametrize('value', ['1', 'true', 'TRUE'])
    def test_fp32_env_yields_float32(self, value):
        """KLPIPE_FP32 truthy: JAX stays at its float32 default."""
        result = _run_python(
            "import kl_pipe; import jax, jax.numpy as jnp; "
            "assert not jax.config.jax_enable_x64; "
            "print(jnp.asarray(1.0).dtype)",
            extra_env={'KLPIPE_FP32': value},
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == 'float32'

    @pytest.mark.parametrize('value', ['', '0', 'false'])
    def test_falsy_env_yields_float64(self, value):
        """Explicitly falsy KLPIPE_FP32 keeps the float64 default."""
        result = _run_python(
            "import kl_pipe; import jax.numpy as jnp; " "print(jnp.asarray(1.0).dtype)",
            extra_env={'KLPIPE_FP32': value},
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == 'float64'


class TestInvalidValue:
    def test_invalid_env_value_raises_in_helper(self, monkeypatch):
        """fp32_requested() raises ValueError on unrecognized values."""
        from kl_pipe._precision import fp32_requested

        monkeypatch.setenv('KLPIPE_FP32', 'banana')
        with pytest.raises(ValueError, match='KLPIPE_FP32'):
            fp32_requested()

    def test_invalid_env_value_fails_import(self):
        """Package import fails loudly on an unrecognized KLPIPE_FP32."""
        result = _run_python(
            "import kl_pipe",
            extra_env={'KLPIPE_FP32': '2'},
        )
        assert result.returncode != 0
        assert 'ValueError' in result.stderr
        assert 'KLPIPE_FP32' in result.stderr


class TestMatmulPrecision:
    def test_matmul_precision_pinned_highest(self):
        """ensure_precision pins matmul precision to 'highest' (forbids the
        silent tf32 tensor-core downgrade of float32 matmuls on GPU)."""
        import jax

        import kl_pipe  # noqa: F401

        assert str(jax.config.jax_default_matmul_precision) == 'highest'

    def test_matmul_precision_pinned_under_fp32(self):
        result = _run_python(
            "import kl_pipe; import jax; "
            "print(jax.config.jax_default_matmul_precision)",
            extra_env={'KLPIPE_FP32': '1'},
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == 'highest'


class TestConftestRespectsPrecision:
    """tests/conftest.py must defer to ensure_precision, never force x64.

    Guards the historical footgun: a hard-coded x64 override in conftest
    silently turned intended-fp32 pytest runs into fp64 ones.
    """

    def test_conftest_import_honors_fp32(self):
        result = _run_python(
            "import tests.conftest; import jax.numpy as jnp; "
            "print(jnp.asarray(1.0).dtype)",
            extra_env={'KLPIPE_FP32': '1'},
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == 'float32'

    def test_conftest_import_default_is_float64(self):
        result = _run_python(
            "import tests.conftest; import jax.numpy as jnp; "
            "print(jnp.asarray(1.0).dtype)",
            extra_env={'KLPIPE_FP32': ''},
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == 'float64'


class TestIdempotence:
    def test_ensure_precision_repeat_call_is_noop(self):
        """Repeat ensure_precision() calls never flip the configured mode."""
        import jax

        from kl_pipe._precision import ensure_precision

        ensure_precision()
        before = jax.config.jax_enable_x64
        ensure_precision()
        assert jax.config.jax_enable_x64 == before
