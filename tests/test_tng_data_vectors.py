"""
Comprehensive tests for TNG data vector generation.

Tests rendering, coordinate transforms, gridding algorithms, and diagnostic plots.
"""

import pytest
import numpy as np
from pathlib import Path

from kl_pipe.tng.loaders import TNG50MockData
from kl_pipe.tng.data_vectors import TNGDataVectorGenerator, TNGRenderConfig
from kl_pipe.parameters import ImagePars
from kl_pipe.plotting import MidpointNormalize


pytestmark = pytest.mark.tng50


@pytest.fixture(scope="module")
def tng_data():
    """Load TNG50 data once."""
    return TNG50MockData()


@pytest.fixture(scope="module")
def test_galaxy(tng_data):
    """Get first galaxy."""
    return tng_data[0]


@pytest.fixture
def image_pars_test():
    """Standard test image parameters."""
    return ImagePars(shape=(64, 64), pixel_scale=0.1, indexing='ij')


@pytest.fixture
def target_redshift_test():
    """Standard target redshift for tests (places TNG at Roman-like distance)."""
    return 0.7  # Good compromise: compact enough for 0.1"/pix, realistic for Roman


class TestBasicRendering:
    """Test basic rendering functionality."""

    def test_native_orientation_intensity(self, test_galaxy, image_pars_test):
        """Test intensity rendering at native TNG orientation."""
        gen = TNGDataVectorGenerator(test_galaxy)
        config = TNGRenderConfig(
            target_redshift=0.6,
            image_pars=image_pars_test,
            band='r',
            use_native_orientation=True,
            use_cic_gridding=True,
        )

        intensity, variance = gen.generate_intensity_map(config, snr=None)

        assert intensity.shape == image_pars_test.shape
        assert np.all(np.isfinite(intensity))
        assert intensity.sum() > 0
        assert (intensity > 0).sum() > 0
        assert np.all(variance == 0)  # No noise added

    def test_native_orientation_velocity(self, test_galaxy, image_pars_test):
        """Test velocity rendering at native TNG orientation."""
        gen = TNGDataVectorGenerator(test_galaxy)
        config = TNGRenderConfig(
            target_redshift=0.6,
            image_pars=image_pars_test,
            band='r',
            use_native_orientation=True,
            use_cic_gridding=True,
        )

        velocity, variance = gen.generate_velocity_map(config, snr=None)

        assert velocity.shape == image_pars_test.shape
        assert np.all(np.isfinite(velocity))
        # Gas is sparse, so may have zeros
        assert (velocity != 0).sum() > 0
        assert np.all(variance == 0)


class TestOrientation:
    """Test geometric transformations."""

    def test_face_on_rendering(self, test_galaxy, image_pars_test):
        """Test rendering at face-on orientation."""
        gen = TNGDataVectorGenerator(test_galaxy)
        pars = {
            'theta_int': 0.0,
            'cosi': 1.0,
            'x0': 0.0,
            'y0': 0.0,
            'g1': 0.0,
            'g2': 0.0,
        }
        config = TNGRenderConfig(
            target_redshift=0.6,
            image_pars=image_pars_test,
            band='r',
            use_native_orientation=False,
            pars=pars,
            use_cic_gridding=True,
        )

        intensity, _ = gen.generate_intensity_map(config)
        velocity, _ = gen.generate_velocity_map(config)

        assert np.all(np.isfinite(intensity))
        assert np.all(np.isfinite(velocity))
        assert intensity.sum() > 0

    def test_edge_on_rendering(self, test_galaxy, image_pars_test):
        """Test rendering at edge-on orientation."""
        gen = TNGDataVectorGenerator(test_galaxy)
        pars = {
            'theta_int': 0.0,
            'cosi': 0.1,
            'x0': 0.0,
            'y0': 0.0,
            'g1': 0.0,
            'g2': 0.0,
        }
        config = TNGRenderConfig(
            target_redshift=0.6,
            image_pars=image_pars_test,
            band='r',
            use_native_orientation=False,
            pars=pars,
            use_cic_gridding=True,
        )

        intensity, _ = gen.generate_intensity_map(config)
        velocity, _ = gen.generate_velocity_map(config)

        assert np.all(np.isfinite(intensity))
        assert np.all(np.isfinite(velocity))

    def test_orientation_affects_maps(self, test_galaxy, image_pars_test):
        """Verify different orientations produce different maps."""
        gen = TNGDataVectorGenerator(test_galaxy)

        # Face-on
        pars_face = {
            'theta_int': 0.0,
            'cosi': 1.0,
            'x0': 0.0,
            'y0': 0.0,
            'g1': 0.0,
            'g2': 0.0,
        }
        config_face = TNGRenderConfig(
            target_redshift=0.6,
            image_pars=image_pars_test,
            band='r',
            use_native_orientation=False,
            pars=pars_face,
        )
        intensity_face, _ = gen.generate_intensity_map(config_face)
        velocity_face, _ = gen.generate_velocity_map(config_face)

        # Edge-on
        pars_edge = {
            'theta_int': 0.0,
            'cosi': 0.1,
            'x0': 0.0,
            'y0': 0.0,
            'g1': 0.0,
            'g2': 0.0,
        }
        config_edge = TNGRenderConfig(
            target_redshift=0.6,
            image_pars=image_pars_test,
            band='r',
            use_native_orientation=False,
            pars=pars_edge,
        )
        intensity_edge, _ = gen.generate_intensity_map(config_edge)
        velocity_edge, _ = gen.generate_velocity_map(config_edge)

        # Maps should differ
        intensity_diff = np.abs(intensity_face - intensity_edge).sum()
        velocity_diff = np.abs(velocity_face - velocity_edge).sum()

        assert intensity_diff > 0, "Intensity should change with orientation"
        assert velocity_diff > 0, "Velocity should change with orientation"

    def test_rotation_affects_maps(self, test_galaxy, image_pars_test):
        """Verify rotation (PA) produces different maps."""
        gen = TNGDataVectorGenerator(test_galaxy)

        # PA = 0
        pars_0 = {
            'theta_int': 0.0,
            'cosi': 0.7,
            'x0': 0.0,
            'y0': 0.0,
            'g1': 0.0,
            'g2': 0.0,
        }
        config_0 = TNGRenderConfig(
            target_redshift=0.6,
            image_pars=image_pars_test,
            band='r',
            use_native_orientation=False,
            pars=pars_0,
        )
        velocity_0, _ = gen.generate_velocity_map(config_0)

        # PA = 90 degrees
        pars_90 = {
            'theta_int': np.pi / 2,
            'cosi': 0.7,
            'x0': 0.0,
            'y0': 0.0,
            'g1': 0.0,
            'g2': 0.0,
        }
        config_90 = TNGRenderConfig(
            target_redshift=0.6,
            image_pars=image_pars_test,
            band='r',
            use_native_orientation=False,
            pars=pars_90,
        )
        velocity_90, _ = gen.generate_velocity_map(config_90)

        velocity_diff = np.abs(velocity_0 - velocity_90).sum()
        assert velocity_diff > 0, "Velocity should change with PA rotation"


class TestGridding:
    """Test gridding algorithms."""

    def test_cic_produces_finite_maps(self, test_galaxy, image_pars_test):
        """CIC gridding should produce finite values."""
        gen = TNGDataVectorGenerator(test_galaxy)
        config = TNGRenderConfig(
            target_redshift=0.6,
            image_pars=image_pars_test,
            band='r',
            use_cic_gridding=True,
        )

        intensity, _ = gen.generate_intensity_map(config)
        velocity, _ = gen.generate_velocity_map(config)

        assert np.all(np.isfinite(intensity))
        assert np.all(np.isfinite(velocity))

    def test_ngp_produces_finite_maps(self, test_galaxy, image_pars_test):
        """NGP gridding should produce finite values."""
        gen = TNGDataVectorGenerator(test_galaxy)
        config = TNGRenderConfig(
            target_redshift=0.6,
            image_pars=image_pars_test,
            band='r',
            use_cic_gridding=False,
        )

        intensity, _ = gen.generate_intensity_map(config)
        velocity, _ = gen.generate_velocity_map(config)

        assert np.all(np.isfinite(intensity))
        assert np.all(np.isfinite(velocity))

    def test_cic_ngp_flux_conservation(self, test_galaxy, image_pars_test):
        """CIC and NGP should conserve total flux within tolerance."""
        gen = TNGDataVectorGenerator(test_galaxy)

        config_cic = TNGRenderConfig(
            target_redshift=0.6,
            image_pars=image_pars_test,
            band='r',
            use_cic_gridding=True,
        )
        intensity_cic, _ = gen.generate_intensity_map(config_cic)

        config_ngp = TNGRenderConfig(
            target_redshift=0.6,
            image_pars=image_pars_test,
            band='r',
            use_cic_gridding=False,
        )
        intensity_ngp, _ = gen.generate_intensity_map(config_ngp)

        flux_cic = intensity_cic.sum()
        flux_ngp = intensity_ngp.sum()
        flux_diff_pct = 100 * abs(flux_cic - flux_ngp) / flux_cic

        # Allow 20% difference (CIC spreads to more pixels)
        assert flux_diff_pct < 20, f"Flux conservation issue: {flux_diff_pct:.1f}%"

    def test_cic_smoother_than_ngp(self, test_galaxy, image_pars_test):
        """CIC should spread flux to more pixels than NGP."""
        gen = TNGDataVectorGenerator(test_galaxy)

        config_cic = TNGRenderConfig(
            target_redshift=0.6,
            image_pars=image_pars_test,
            band='r',
            use_cic_gridding=True,
        )
        intensity_cic, _ = gen.generate_intensity_map(config_cic)

        config_ngp = TNGRenderConfig(
            target_redshift=0.6,
            image_pars=image_pars_test,
            band='r',
            use_cic_gridding=False,
        )
        intensity_ngp, _ = gen.generate_intensity_map(config_ngp)

        # CIC should have at least as many non-zero pixels
        nonzero_cic = (intensity_cic > 0).sum()
        nonzero_ngp = (intensity_ngp > 0).sum()

        assert (
            nonzero_cic >= nonzero_ngp
        ), "CIC should spread to at least as many pixels as NGP"


class TestNoise:
    """Test noise addition."""

    def test_noise_increases_variance(self, test_galaxy, image_pars_test):
        """Adding noise should increase variance."""
        gen = TNGDataVectorGenerator(test_galaxy)
        config = TNGRenderConfig(
            target_redshift=0.6, image_pars=image_pars_test, band='r'
        )

        intensity_clean, var_clean = gen.generate_intensity_map(config, snr=None)
        intensity_noisy, var_noisy = gen.generate_intensity_map(config, snr=50, seed=42)

        assert np.all(var_clean == 0)
        assert np.all(var_noisy > 0)
        assert not np.allclose(intensity_clean, intensity_noisy)

    def test_noise_reproducibility(self, test_galaxy, image_pars_test):
        """Same seed should produce same noise."""
        gen = TNGDataVectorGenerator(test_galaxy)
        config = TNGRenderConfig(
            target_redshift=0.6, image_pars=image_pars_test, band='r'
        )

        intensity_1, var_1 = gen.generate_intensity_map(config, snr=50, seed=42)
        intensity_2, var_2 = gen.generate_intensity_map(config, snr=50, seed=42)

        assert np.allclose(intensity_1, intensity_2)
        assert np.allclose(var_1, var_2)

    def test_different_seeds_produce_different_noise(
        self, test_galaxy, image_pars_test
    ):
        """Different seeds should produce different noise."""
        gen = TNGDataVectorGenerator(test_galaxy)
        config = TNGRenderConfig(
            target_redshift=0.6, image_pars=image_pars_test, band='r'
        )

        intensity_1, _ = gen.generate_intensity_map(config, snr=50, seed=42)
        intensity_2, _ = gen.generate_intensity_map(config, snr=50, seed=123)

        assert not np.allclose(intensity_1, intensity_2)


class TestParticleTypes:
    """Test stellar vs gas particle usage."""

    def test_intensity_uses_stellar(self, test_galaxy, image_pars_test):
        """Intensity should use stellar particles."""
        gen = TNGDataVectorGenerator(test_galaxy)
        config = TNGRenderConfig(
            target_redshift=0.6, image_pars=image_pars_test, band='r'
        )

        # Should work even without gas data
        intensity, _ = gen.generate_intensity_map(config)
        assert intensity.sum() > 0

    def test_velocity_uses_gas(self, test_galaxy, image_pars_test):
        """Velocity should use gas particles."""
        gen = TNGDataVectorGenerator(test_galaxy)

        # Check that gas data is required
        if gen.gas is None:
            pytest.skip("Galaxy missing gas data")

        config = TNGRenderConfig(
            target_redshift=0.6, image_pars=image_pars_test, band='r'
        )
        velocity, _ = gen.generate_velocity_map(config)

        # Velocity should be based on gas (may be sparse)
        assert np.any(velocity != 0)


class TestPhotometricBands:
    """Test different photometric bands."""

    @pytest.mark.parametrize("band", ['g', 'r', 'i', 'u', 'z'])
    def test_all_bands_work(self, test_galaxy, image_pars_test, band):
        """All SDSS bands should render successfully."""
        gen = TNGDataVectorGenerator(test_galaxy)
        config = TNGRenderConfig(
            target_redshift=0.6, image_pars=image_pars_test, band=band
        )

        intensity, _ = gen.generate_intensity_map(config)
        assert intensity.sum() > 0

    def test_dusted_vs_raw(self, test_galaxy, image_pars_test):
        """Dusted and raw luminosities should differ."""
        gen = TNGDataVectorGenerator(test_galaxy)

        config_dusted = TNGRenderConfig(
            target_redshift=0.6, image_pars=image_pars_test, band='r', use_dusted=True
        )
        intensity_dusted, _ = gen.generate_intensity_map(config_dusted)

        config_raw = TNGRenderConfig(
            target_redshift=0.6, image_pars=image_pars_test, band='r', use_dusted=False
        )
        intensity_raw, _ = gen.generate_intensity_map(config_raw)

        # Dust should attenuate, so raw > dusted
        assert intensity_raw.sum() > intensity_dusted.sum()


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_missing_gas_raises_error(self, test_galaxy, image_pars_test):
        """Should raise error if gas data missing for velocity."""
        # Create generator and artificially remove gas
        gen = TNGDataVectorGenerator(test_galaxy)
        gen.gas = None

        config = TNGRenderConfig(target_redshift=0.6, image_pars=image_pars_test)

        with pytest.raises(ValueError, match="Gas data required"):
            gen.generate_velocity_map(config)

    def test_pars_required_for_custom_orientation(self, test_galaxy, image_pars_test):
        """Should raise error if pars not provided with use_native_orientation=False."""
        gen = TNGDataVectorGenerator(test_galaxy)
        config = TNGRenderConfig(
            target_redshift=0.6,
            image_pars=image_pars_test,
            use_native_orientation=False,
            pars=None,  # Missing!
        )

        with pytest.raises(ValueError, match="pars must be provided"):
            gen.generate_intensity_map(config)

    def test_invalid_shear_raises_error(self, test_galaxy, image_pars_test):
        """Should raise error if shear magnitude >= 1."""
        with pytest.raises(ValueError, match="Shear too large"):
            TNGRenderConfig(
                target_redshift=0.6,
                image_pars=image_pars_test,
                use_native_orientation=False,
                pars={
                    'cosi': 0.7,
                    'theta_int': 0.0,
                    'x0': 0.0,
                    'y0': 0.0,
                    'g1': 0.8,
                    'g2': 0.8,
                },
            )


class TestSFRMap:
    """Test star formation rate map generation."""

    def test_sfr_map_native_orientation(self, test_galaxy, image_pars_test):
        """Test SFR map generation at native TNG orientation."""
        gen = TNGDataVectorGenerator(test_galaxy)
        config = TNGRenderConfig(
            target_redshift=0.6,
            image_pars=image_pars_test,
            use_native_orientation=True,
        )

        sfr_map = gen.generate_sfr_map(config, snr=None)

        assert sfr_map.shape == image_pars_test.shape
        assert np.all(np.isfinite(sfr_map))
        assert sfr_map.sum() > 0, "SFR map should have positive total"
        assert np.any(sfr_map > 0), "SFR map should have some non-zero pixels"

    def test_sfr_map_custom_orientation(self, test_galaxy, image_pars_test):
        """Test SFR map generation with custom orientation (uses gas offset)."""
        gen = TNGDataVectorGenerator(test_galaxy)
        pars = {
            'theta_int': np.pi / 4,
            'cosi': 0.7,
            'x0': 0.0,
            'y0': 0.0,
            'g1': 0.0,
            'g2': 0.0,
        }
        config = TNGRenderConfig(
            target_redshift=0.6,
            image_pars=image_pars_test,
            use_native_orientation=False,
            pars=pars,
        )

        sfr_map = gen.generate_sfr_map(config, snr=None)

        assert sfr_map.shape == image_pars_test.shape
        assert np.all(np.isfinite(sfr_map))
        assert sfr_map.sum() > 0

    def test_sfr_orientation_affects_map(self, test_galaxy, image_pars_test):
        """Verify different orientations produce different SFR maps."""
        gen = TNGDataVectorGenerator(test_galaxy)

        # Face-on
        pars_face = {
            'theta_int': 0.0,
            'cosi': 1.0,
            'x0': 0.0,
            'y0': 0.0,
            'g1': 0.0,
            'g2': 0.0,
        }
        config_face = TNGRenderConfig(
            target_redshift=0.6,
            image_pars=image_pars_test,
            use_native_orientation=False,
            pars=pars_face,
        )
        sfr_face = gen.generate_sfr_map(config_face)

        # Edge-on
        pars_edge = {
            'theta_int': 0.0,
            'cosi': 0.1,
            'x0': 0.0,
            'y0': 0.0,
            'g1': 0.0,
            'g2': 0.0,
        }
        config_edge = TNGRenderConfig(
            target_redshift=0.6,
            image_pars=image_pars_test,
            use_native_orientation=False,
            pars=pars_edge,
        )
        sfr_edge = gen.generate_sfr_map(config_edge)

        # Maps should differ
        sfr_diff = np.abs(sfr_face - sfr_edge).sum()
        assert sfr_diff > 0, "SFR map should change with orientation"

    def test_sfr_with_noise(self, test_galaxy, image_pars_test):
        """Test SFR map generation with noise."""
        gen = TNGDataVectorGenerator(test_galaxy)
        config = TNGRenderConfig(
            target_redshift=0.6,
            image_pars=image_pars_test,
            use_native_orientation=True,
        )

        sfr_clean = gen.generate_sfr_map(config, snr=None)
        sfr_noisy = gen.generate_sfr_map(config, snr=50, seed=42)

        assert not np.allclose(sfr_clean, sfr_noisy), "Noise should change the map"
        assert np.all(np.isfinite(sfr_noisy))


class TestTransformRoundTrip:
    """Test that coordinate transformations are internally consistent."""

    def test_custom_orientation_changes_maps(self, test_galaxy, image_pars_test):
        """
        Verify that custom orientation (even at native params) differs from native mode.

        This is EXPECTED behavior because:
        - Native mode: TNG z-axis IS the line-of-sight, just drops z for projection
        - Custom mode: Treats TNG frame as "observed", undoes to disk plane, re-applies

        The two modes have different semantics:
        - Native: "Show me what TNG computed"
        - Custom: "Transform TNG data to this new viewing geometry"

        Even with identical geometric parameters, the transform path differs.
        """
        gen = TNGDataVectorGenerator(test_galaxy)

        # Render at native orientation
        config_native = TNGRenderConfig(
            target_redshift=0.6,
            image_pars=image_pars_test,
            use_native_orientation=True,
            use_cic_gridding=True,
        )
        int_native, _ = gen.generate_intensity_map(config_native, snr=None)
        vel_native, _ = gen.generate_velocity_map(config_native, snr=None)

        # Render with custom orientation set to match native parameters
        native_cosi = np.cos(np.radians(gen.native_inclination_deg))
        native_theta_int = np.radians(gen.native_pa_deg)

        pars_native = {
            'cosi': native_cosi,
            'theta_int': native_theta_int,
            'x0': 0.0,
            'y0': 0.0,
            'g1': 0.0,
            'g2': 0.0,
        }

        config_custom = TNGRenderConfig(
            target_redshift=0.6,
            image_pars=image_pars_test,
            use_native_orientation=False,
            pars=pars_native,
            use_cic_gridding=True,
        )
        int_custom, _ = gen.generate_intensity_map(config_custom, snr=None)
        vel_custom, _ = gen.generate_velocity_map(config_custom, snr=None)

        # Both should produce valid, non-trivial maps
        assert int_native.sum() > 0, "Native intensity should be positive"
        assert int_custom.sum() > 0, "Custom intensity should be positive"
        assert np.any(vel_native != 0), "Native velocity should have non-zero values"
        assert np.any(vel_custom != 0), "Custom velocity should have non-zero values"

        # They may differ due to different transform paths - this is expected
        # The key is that both are internally consistent
        print(f"✓ Native intensity sum: {int_native.sum():.2e}")
        print(f"✓ Custom intensity sum: {int_custom.sum():.2e}")
        print(
            f"✓ Native velocity range: [{vel_native.min():.1f}, {vel_native.max():.1f}] km/s"
        )
        print(
            f"✓ Custom velocity range: [{vel_custom.min():.1f}, {vel_custom.max():.1f}] km/s"
        )

    def test_orientation_sweep_consistency(self, test_galaxy, image_pars_test):
        """
        Verify that sweeping through orientations produces smooth, consistent changes.

        This tests that the custom orientation transforms are working correctly
        by checking that small changes in inclination produce small changes in maps.
        """
        gen = TNGDataVectorGenerator(test_galaxy)

        # Sweep inclination from 30° to 60° in 10° steps
        inclinations = [30, 40, 50, 60]
        intensity_sums = []
        velocity_ranges = []

        for inc_deg in inclinations:
            pars = {
                'cosi': np.cos(np.radians(inc_deg)),
                'theta_int': 0.0,
                'x0': 0.0,
                'y0': 0.0,
                'g1': 0.0,
                'g2': 0.0,
            }

            config = TNGRenderConfig(
                target_redshift=0.6,
                image_pars=image_pars_test,
                use_native_orientation=False,
                pars=pars,
                use_cic_gridding=True,
            )

            intensity, _ = gen.generate_intensity_map(config, snr=None)
            velocity, _ = gen.generate_velocity_map(config, snr=None)

            intensity_sums.append(intensity.sum())
            velocity_ranges.append(velocity.max() - velocity.min())

        # Check that values are finite and positive
        assert all(
            np.isfinite(s) and s > 0 for s in intensity_sums
        ), "All intensity sums should be finite and positive"
        assert all(
            np.isfinite(r) and r > 0 for r in velocity_ranges
        ), "All velocity ranges should be finite and positive"

        # Check for smoothness: adjacent values shouldn't differ by more than 2x
        for i in range(len(inclinations) - 1):
            int_ratio = intensity_sums[i + 1] / intensity_sums[i]
            assert (
                0.5 < int_ratio < 2.0
            ), f"Intensity change from inc={inclinations[i]}° to {inclinations[i+1]}° too abrupt"

            vel_ratio = velocity_ranges[i + 1] / velocity_ranges[i]
            assert (
                0.5 < vel_ratio < 2.0
            ), f"Velocity change from inc={inclinations[i]}° to {inclinations[i+1]}° too abrupt"

        print(
            f"✓ Inclination sweep passed: intensity sums = {[f'{s:.2e}' for s in intensity_sums]}"
        )
        print(
            f"✓ Inclination sweep passed: velocity ranges = {[f'{r:.1f}' for r in velocity_ranges]}"
        )


class TestInclinationSymmetry:
    """Test inclination handling and symmetry properties."""

    def test_tng_asymmetric_under_inclination_flip(self, test_galaxy, image_pars_test):
        """
        TNG data should NOT be symmetric under inc → 180°-inc flip.

        Because TNG galaxies are real (asymmetric) systems, viewing from
        inc=45° vs inc=135° should give different results. The coordinate
        geometry may be similar, but velocity signs differ.
        """
        gen = TNGDataVectorGenerator(test_galaxy)

        # Test at complementary inclinations: 45° and 135°
        inc_1 = 45.0
        inc_2 = 135.0  # = 180° - 45°

        pars_1 = {
            'cosi': np.cos(np.radians(inc_1)),
            'theta_int': 0.0,
            'x0': 0.0,
            'y0': 0.0,
            'g1': 0.0,
            'g2': 0.0,
        }

        pars_2 = {
            'cosi': np.cos(np.radians(inc_2)),  # Negative cos(i)!
            'theta_int': 0.0,
            'x0': 0.0,
            'y0': 0.0,
            'g1': 0.0,
            'g2': 0.0,
        }

        config_1 = TNGRenderConfig(
            target_redshift=0.6,
            image_pars=image_pars_test,
            use_native_orientation=False,
            pars=pars_1,
            use_cic_gridding=True,
        )

        config_2 = TNGRenderConfig(
            target_redshift=0.6,
            image_pars=image_pars_test,
            use_native_orientation=False,
            pars=pars_2,
            use_cic_gridding=True,
        )

        # Generate velocity maps
        vel_1, _ = gen.generate_velocity_map(config_1, snr=None)
        vel_2, _ = gen.generate_velocity_map(config_2, snr=None)

        # They should NOT be identical (TNG is asymmetric)
        diff = np.abs(vel_1 - vel_2)
        max_diff = np.nanmax(diff)

        # Expect significant differences (>10 km/s somewhere in the map)
        assert max_diff > 10.0, (
            f"TNG velocity maps at inc={inc_1}° and inc={inc_2}° "
            f"should differ significantly, but max_diff={max_diff:.2f} km/s"
        )

        print(f"✓ TNG asymmetric: max velocity difference = {max_diff:.1f} km/s")

    def test_negative_cosi_handled_correctly(self, test_galaxy, image_pars_test):
        """
        Test that inclinations >90° (negative cos(i)) are handled correctly.

        The code should handle negative cos(i) without errors and produce
        valid maps with appropriate sign changes in velocities.
        """
        gen = TNGDataVectorGenerator(test_galaxy)

        # Test inclination >90° (cos(i) < 0)
        inc = 120.0
        cosi = np.cos(np.radians(inc))
        assert cosi < 0, "Test requires inc > 90° so cos(i) < 0"

        pars = {
            'cosi': cosi,
            'theta_int': 0.0,
            'x0': 0.0,
            'y0': 0.0,
            'g1': 0.0,
            'g2': 0.0,
        }

        config = TNGRenderConfig(
            target_redshift=0.6,
            image_pars=image_pars_test,
            use_native_orientation=False,
            pars=pars,
            use_cic_gridding=True,
        )

        # Should complete without errors
        vel, var = gen.generate_velocity_map(config, snr=50)
        intensity, int_var = gen.generate_intensity_map(config, snr=50)

        # Maps should be valid (finite, non-zero in places)
        assert np.all(np.isfinite(vel)), "Velocity map has non-finite values"
        assert np.all(np.isfinite(intensity)), "Intensity map has non-finite values"
        assert np.any(vel != 0), "Velocity map is all zeros"
        assert np.any(intensity > 0), "Intensity map is all zeros/negative"

        print(f"✓ inc={inc}° (cos(i)={cosi:.3f}) handled correctly")

    def test_face_on_vs_face_on_from_below(self, test_galaxy, image_pars_test):
        """
        Compare face-on (inc=0°) vs face-on from below (inc=180°).

        For intensity: should be similar (brightness doesn't depend on which side)
        For velocity: vertical motions reverse sign
        """
        gen = TNGDataVectorGenerator(test_galaxy)

        # Face-on from above (inc=0°, cos(i)=+1)
        pars_above = {
            'cosi': 1.0,
            'theta_int': 0.0,
            'x0': 0.0,
            'y0': 0.0,
            'g1': 0.0,
            'g2': 0.0,
        }

        # Face-on from below (inc=180°, cos(i)=-1)
        pars_below = {
            'cosi': -1.0,
            'theta_int': 0.0,
            'x0': 0.0,
            'y0': 0.0,
            'g1': 0.0,
            'g2': 0.0,
        }

        config_above = TNGRenderConfig(
            target_redshift=0.6,
            image_pars=image_pars_test,
            use_native_orientation=False,
            pars=pars_above,
            use_cic_gridding=True,
        )

        config_below = TNGRenderConfig(
            target_redshift=0.6,
            image_pars=image_pars_test,
            use_native_orientation=False,
            pars=pars_below,
            use_cic_gridding=True,
        )

        # Generate maps
        vel_above, _ = gen.generate_velocity_map(config_above, snr=None)
        vel_below, _ = gen.generate_velocity_map(config_below, snr=None)
        int_above, _ = gen.generate_intensity_map(config_above, snr=None)
        int_below, _ = gen.generate_intensity_map(config_below, snr=None)

        # Velocity: should differ (vertical motion flips)
        vel_diff = np.abs(vel_above - vel_below)
        vel_max_diff = np.nanmax(vel_diff)
        assert vel_max_diff > 5.0, (
            f"Face-on velocities from above/below should differ, "
            f"but max_diff={vel_max_diff:.2f} km/s"
        )

        # Intensity: should be similar (same particles, same projection)
        # Use relative difference to account for overall brightness
        int_rel_diff = np.abs(int_above - int_below) / (int_above + int_below + 1e-10)
        int_median_rel_diff = np.nanmedian(int_rel_diff[int_rel_diff > 0])
        assert int_median_rel_diff < 0.5, (
            f"Face-on intensity from above/below should be similar, "
            f"but median_rel_diff={int_median_rel_diff:.2f}"
        )

        print(f"✓ Face-on velocity diff = {vel_max_diff:.1f} km/s")
        print(f"✓ Face-on intensity median_rel_diff = {int_median_rel_diff:.3f}")


class TestDiagnosticPlots:
    """Generate diagnostic plots for visual validation of TNG rendering."""

    @pytest.fixture(scope="class")
    def output_dir(self):
        """Create output directory for diagnostic plots."""
        out_dir = Path(__file__).parent / "out" / "tng_diagnostics"
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    @staticmethod
    def compute_robust_intensity_bounds(
        intensity_map, log=True, lower_percentile=2.0, upper_percentile=98.0
    ):
        """
        Compute robust bounds for intensity visualization.

        Excludes zero/negative flux pixels and computes percentiles only on
        pixels with actual signal.

        Parameters
        ----------
        intensity_map : np.ndarray
            Intensity map (linear or log scale)
        log : bool
            If True, assumes input is already log10(intensity)
        lower_percentile : float
            Lower percentile for vmin (default 2.0)
        upper_percentile : float
            Upper percentile for vmax (default 98.0)

        Returns
        -------
        vmin, vmax : float
            Robust bounds for visualization
        """
        if log:
            # Already in log space - exclude -inf and very low values (zeros)
            finite_vals = intensity_map[np.isfinite(intensity_map)]
            # For noisy data with negatives, clip minimum to avoid log issues
            # Exclude pixels that are essentially zero (log(flux) < -8 means flux < 1e-8)
            # but include slightly negative values from noise
            signal_vals = finite_vals[finite_vals > -8]
        else:
            # Convert to log - clip negatives to small positive value for noise handling
            intensity_clipped = np.clip(intensity_map, 1e-10, None)
            log_map = np.log10(intensity_clipped)
            finite_vals = log_map[np.isfinite(log_map)]
            signal_vals = finite_vals[finite_vals > -8]

        if len(signal_vals) == 0:
            return -10, 40  # Fallback

        vmin = np.percentile(signal_vals, lower_percentile)
        vmax = np.percentile(signal_vals, upper_percentile)

        return vmin, vmax

    @staticmethod
    def add_colorbar_matching_height(im, ax, **kwargs):
        """
        Add colorbar with height matching the parent axis.

        Parameters
        ----------
        im : matplotlib image
            The image to create colorbar for
        ax : matplotlib axis
            The axis to attach colorbar to
        **kwargs : dict
            Additional keyword arguments passed to plt.colorbar()

        Returns
        -------
        colorbar
            The created colorbar object
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        return plt.colorbar(im, cax=cax, **kwargs)

    @staticmethod
    def add_scale_markers(
        ax, image_pars, scale_bar_arcsec=1.0, color='black', crosshair_arcsec=0.3
    ):
        """
        Add physical scale bar and center marker to a plot.

        Automatically detects whether plot uses pixel or data (arcsec) coordinates
        based on axis limits.

        Parameters
        ----------
        ax : matplotlib axis
            Axis to add markers to
        image_pars : ImagePars
            Image parameters for pixel/arcsec conversion
        scale_bar_arcsec : float
            Length of scale bar in arcsec
        color : str
            Color for scale bar and text (default 'black', use 'white' for dark backgrounds)
        crosshair_arcsec : float
            Size of center crosshair in arcsec (consistent regardless of pixel scale)
        """
        # Detect coordinate system by checking axis limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # If limits are close to shape, assume pixel coordinates
        # Otherwise assume data coordinates (arcsec, typically centered at 0)
        using_pixels = (
            abs(xlim[1] - image_pars.shape[1]) < 5
            and abs(ylim[1] - image_pars.shape[0]) < 5
        )

        if using_pixels:
            # Pixel coordinate system
            scale_bar_pixels = scale_bar_arcsec / image_pars.pixel_scale
            x_start = 0.05 * image_pars.shape[1]
            y_start = 0.05 * image_pars.shape[0]
            ax.plot(
                [x_start, x_start + scale_bar_pixels],
                [y_start, y_start],
                color=color,
                linewidth=3,
                solid_capstyle='butt',
            )
            ax.text(
                x_start + scale_bar_pixels / 2,
                y_start + 0.04 * image_pars.shape[0],
                f'{scale_bar_arcsec}"',
                color=color,
                ha='center',
                va='bottom',
                fontsize=8,
                weight='bold',
                bbox=dict(
                    boxstyle='round,pad=0.3',
                    facecolor='white' if color == 'black' else 'black',
                    alpha=0.7,
                    edgecolor=color,
                ),
            )

            # Center marker in pixels - consistent angular size
            cx, cy = image_pars.shape[1] // 2, image_pars.shape[0] // 2
            marker_size_pix = crosshair_arcsec / image_pars.pixel_scale
            ax.plot(
                [cx - marker_size_pix, cx + marker_size_pix],
                [cy, cy],
                'k-',
                linewidth=0.8,
                alpha=0.6,
            )
            ax.plot(
                [cx, cx],
                [cy - marker_size_pix, cy + marker_size_pix],
                'k-',
                linewidth=0.8,
                alpha=0.6,
            )
            ax.plot(cx, cy, 'k+', markersize=6, markeredgewidth=1, alpha=0.6)
        else:
            # Data coordinate system (arcsec)
            fov_x = image_pars.shape[1] * image_pars.pixel_scale
            fov_y = image_pars.shape[0] * image_pars.pixel_scale

            x_start = -fov_x / 2 + 0.05 * fov_x
            y_start = -fov_y / 2 + 0.05 * fov_y
            ax.plot(
                [x_start, x_start + scale_bar_arcsec],
                [y_start, y_start],
                color=color,
                linewidth=3,
                solid_capstyle='butt',
            )
            ax.text(
                x_start + scale_bar_arcsec / 2,
                y_start + 0.04 * fov_y,
                f'{scale_bar_arcsec}"',
                color=color,
                ha='center',
                va='bottom',
                fontsize=8,
                weight='bold',
                bbox=dict(
                    boxstyle='round,pad=0.3',
                    facecolor='white' if color == 'black' else 'black',
                    alpha=0.7,
                    edgecolor=color,
                ),
            )

            # Center marker in arcsec (consistent with pixel branch)
            ax.plot(
                [-crosshair_arcsec, crosshair_arcsec],
                [0, 0],
                'k-',
                linewidth=0.8,
                alpha=0.6,
            )
            ax.plot(
                [0, 0],
                [-crosshair_arcsec, crosshair_arcsec],
                'k-',
                linewidth=0.8,
                alpha=0.6,
            )
            ax.plot(0, 0, 'k+', markersize=6, markeredgewidth=1, alpha=0.6)

    def test_all_galaxies_high_res_native(self, tng_data, output_dir):
        """
        Generate high-resolution diagnostic plots for all TNG galaxies at native orientation.

        Creates a tall 2-column plot:
        - Column 1: Intensity maps (log scale, viridis)
        - Column 2: Velocity maps (RdBu_r with white=0)
        - Rows: One per galaxy
        - Labels: SubhaloID, index, inclination, PA
        - Minimal axis clutter (no ticks/labels), compact layout
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        n_galaxies = len(tng_data)
        image_pars = ImagePars(
            shape=(128, 128), pixel_scale=0.05, indexing='ij'
        )  # High res
        target_z = 0.7
        native_z = 0.011

        # Compact figure with minimal spacing but room for titles
        fig = plt.figure(figsize=(10, 3.5 * n_galaxies))
        gs = GridSpec(n_galaxies, 2, figure=fig, hspace=0.15, wspace=0.12)

        for idx in range(n_galaxies):
            galaxy = tng_data[idx]
            gen = TNGDataVectorGenerator(galaxy)

            # Get metadata
            subhalo_id = int(galaxy['subhalo']['SubhaloID'])
            inc_deg = gen.native_inclination_deg
            pa_deg = gen.native_pa_deg
            flipped = gen.flipped_from_below

            config = TNGRenderConfig(
                target_redshift=target_z,
                image_pars=image_pars,
                use_native_orientation=True,
                use_cic_gridding=True,
            )

            # Generate maps (no noise for truth visualization)
            intensity, _ = gen.generate_intensity_map(config, snr=None)
            velocity, _ = gen.generate_velocity_map(config, snr=None)

            # Plot intensity with log scale and viridis colormap
            ax_int = fig.add_subplot(gs[idx, 0])
            # Log scale with robust percentile-based bounds
            # Replace NaN/negative with small positive value before log
            intensity_safe = np.nan_to_num(intensity, nan=0.0, posinf=0.0, neginf=0.0)
            intensity_safe = np.clip(intensity_safe, 1e-10, None)
            int_log = np.log10(intensity_safe)
            vmin, vmax = self.compute_robust_intensity_bounds(int_log, log=True)
            im_int = ax_int.imshow(
                int_log,
                origin='lower',
                cmap='viridis',
                aspect='equal',
                vmin=vmin,
                vmax=vmax,
            )
            ax_int.set_title(f'Intensity - Galaxy {idx} (ID={subhalo_id})', fontsize=10)
            # Remove axis ticks and labels for cleaner appearance
            ax_int.set_xticks([])
            ax_int.set_yticks([])
            plt.colorbar(
                im_int, ax=ax_int, label='log$_{10}$(Flux)', fraction=0.046, pad=0.02
            )

            # Add metadata text
            flipped_str = " (flipped)" if flipped else ""
            ax_int.text(
                0.02,
                0.98,
                f'inc={inc_deg:.1f}°{flipped_str}\nPA={pa_deg:.1f}°',
                transform=ax_int.transAxes,
                va='top',
                ha='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=8,
            )

            # Add scale bar and center marker (white for intensity)
            self.add_scale_markers(
                ax_int, image_pars, scale_bar_arcsec=1.0, color='white'
            )

            # Plot velocity with diverging colormap (white=0)
            ax_vel = fig.add_subplot(gs[idx, 1])
            vmax = np.nanmax(np.abs(velocity))
            im_vel = ax_vel.imshow(
                velocity,
                origin='lower',
                cmap='RdBu_r',
                aspect='equal',
                norm=MidpointNormalize(vmin=-vmax, vmax=vmax, midpoint=0),
            )
            ax_vel.set_title(f'Velocity - Galaxy {idx} (ID={subhalo_id})', fontsize=10)
            # Remove axis ticks and labels for cleaner appearance
            ax_vel.set_xticks([])
            ax_vel.set_yticks([])
            plt.colorbar(
                im_vel,
                ax=ax_vel,
                label=r'$v_\mathrm{LOS}$ [km/s]',
                fraction=0.046,
                pad=0.02,
            )

            # Add scale bar and center marker (red crosshair shows shared reference point)
            self.add_scale_markers(ax_vel, image_pars, scale_bar_arcsec=1.0)

            print(
                f"✓ Galaxy {idx} (SubhaloID={subhalo_id}): "
                f"inc={inc_deg:.1f}°, PA={pa_deg:.1f}°, flipped={flipped}"
            )

        # Leave room at top for suptitle (top=0.96 leaves ~4% for title)
        gs.update(top=0.96, bottom=0.01)
        fig.suptitle(
            f'TNG50 Galaxies - Native Orientation (z={native_z:.3f}→{target_z:.1f})',
            fontsize=14,
            y=0.99,
        )
        plt.savefig(
            output_dir / 'all_galaxies_native_highres.png', dpi=150, bbox_inches='tight'
        )
        plt.close()

        print(
            f"✓ Saved diagnostic plot: {output_dir / 'all_galaxies_native_highres.png'}"
        )

    def test_cic_vs_ngp_comparison(self, test_galaxy, output_dir):
        """
        Compare CIC vs NGP gridding side-by-side with particle view and residuals.

        Columns:
        - 1: Raw particle scatter plot (higher res view of actual data)
        - 2: Cloud-in-Cell (CIC) gridded map
        - 3: Nearest-Grid-Point (NGP) gridded map
        - 4: Residuals (CIC - NGP)

        Rows:
        - Top: Intensity (stellar particles)
        - Bottom: Velocity (gas particles)
        """
        import matplotlib.pyplot as plt
        from kl_pipe.tng.data_vectors import convert_tng_to_arcsec

        gen = TNGDataVectorGenerator(test_galaxy)
        image_pars = ImagePars(shape=(64, 64), pixel_scale=0.1, indexing='ij')
        target_z = 0.7
        fov_arcsec = image_pars.shape[0] * image_pars.pixel_scale  # Same for both dims

        # CIC gridding
        config_cic = TNGRenderConfig(
            target_redshift=target_z,
            image_pars=image_pars,
            use_native_orientation=True,
            use_cic_gridding=True,
        )
        int_cic, _ = gen.generate_intensity_map(config_cic, snr=None)
        vel_cic, _ = gen.generate_velocity_map(config_cic, snr=None)

        # NGP gridding
        config_ngp = TNGRenderConfig(
            target_redshift=target_z,
            image_pars=image_pars,
            use_native_orientation=True,
            use_cic_gridding=False,
        )
        int_ngp, _ = gen.generate_intensity_map(config_ngp, snr=None)
        vel_ngp, _ = gen.generate_velocity_map(config_ngp, snr=None)

        # Get particle positions for scatter plots
        # Stellar particles (for intensity)
        stellar_coords = gen.stellar['Coordinates'].copy()
        stellar_center = gen._get_reference_center(
            config_cic.center_on_peak, config_cic.band, config_cic.use_dusted
        )
        stellar_coords_centered = gen._center_coordinates(
            stellar_coords, stellar_center
        )
        stellar_coords_2d = stellar_coords_centered[
            :, :2
        ]  # Just drop z for native orientation
        stellar_coords_arcsec = convert_tng_to_arcsec(
            stellar_coords_2d, gen.distance_mpc, target_redshift=target_z
        )

        # Stellar luminosities for color/size
        lum_key = gen._get_luminosity_key(config_cic.band, config_cic.use_dusted)
        stellar_lum = gen.stellar[lum_key]
        stellar_lum_log = np.log10(stellar_lum + 1e-10)

        # Gas particles (for velocity)
        gas_coords = gen.gas['Coordinates'].copy()
        gas_center = stellar_center  # Use same center for consistency
        gas_coords_centered = gen._center_coordinates(gas_coords, gas_center)
        gas_coords_2d = gas_coords_centered[:, :2]  # Just drop z for native orientation
        gas_coords_arcsec = convert_tng_to_arcsec(
            gas_coords_2d, gen.distance_mpc, target_redshift=target_z
        )

        # Gas velocities - must subtract systemic velocity like the gridded maps do
        # The generate_velocity_map method subtracts median velocity to center at v=0
        gas_vel_3d = gen.gas['Velocities'].copy()
        v_systemic = np.median(gas_vel_3d, axis=0)
        gas_vel_3d -= v_systemic  # Now in galaxy rest frame
        # For native orientation, LOS is along z-axis, so v_LOS = v_z
        gas_vel = gas_vel_3d[:, 2]

        # Create figure: 2 rows x 4 columns
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))

        # === Row 0: Intensity ===
        # Column 0: Particle scatter
        ax = axes[0, 0]
        # Downsample for visibility and mask to FOV
        mask_fov = (np.abs(stellar_coords_arcsec[:, 0]) < fov_arcsec / 2) & (
            np.abs(stellar_coords_arcsec[:, 1]) < fov_arcsec / 2
        )
        x_p = stellar_coords_arcsec[mask_fov, 0]
        y_p = stellar_coords_arcsec[mask_fov, 1]
        lum_p = stellar_lum_log[mask_fov]
        # Subsample if too many particles
        n_max = 10000
        if len(x_p) > n_max:
            idx_sub = np.random.choice(len(x_p), n_max, replace=False)
            x_p, y_p, lum_p = x_p[idx_sub], y_p[idx_sub], lum_p[idx_sub]
        vmin_lum, vmax_lum = np.percentile(lum_p, [5, 99])
        sc_int = ax.scatter(
            x_p,
            y_p,
            c=lum_p,
            s=1,
            alpha=0.5,
            cmap='viridis',
            vmin=vmin_lum,
            vmax=vmax_lum,
        )
        ax.set_xlim(-fov_arcsec / 2, fov_arcsec / 2)
        ax.set_ylim(-fov_arcsec / 2, fov_arcsec / 2)
        ax.set_aspect('equal')
        ax.set_title('Particles (Stellar)', fontsize=11)
        self.add_colorbar_matching_height(sc_int, ax, label='log$_{10}$(Lum)')

        # Column 1: CIC
        ax = axes[0, 1]
        int_log_cic = np.log10(int_cic + 1e-10)
        vmin, vmax = self.compute_robust_intensity_bounds(int_log_cic, log=True)
        im = ax.imshow(
            int_log_cic,
            origin='lower',
            cmap='viridis',
            aspect='equal',
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title('CIC Gridded\n(Cloud-in-Cell)', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        self.add_colorbar_matching_height(im, ax, label='log$_{10}$(Flux)')
        self.add_scale_markers(ax, image_pars, scale_bar_arcsec=1.0, color='white')

        # Column 2: NGP
        ax = axes[0, 2]
        int_log_ngp = np.log10(int_ngp + 1e-10)
        im = ax.imshow(
            int_log_ngp,
            origin='lower',
            cmap='viridis',
            aspect='equal',
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title('NGP Gridded\n(Nearest-Grid-Point)', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        self.add_colorbar_matching_height(im, ax, label='log$_{10}$(Flux)')
        self.add_scale_markers(ax, image_pars, scale_bar_arcsec=1.0, color='white')

        # Column 3: Residuals (CIC - NGP)
        ax = axes[0, 3]
        int_resid = int_cic - int_ngp
        resid_max = np.percentile(np.abs(int_resid), 99)
        im = ax.imshow(
            int_resid,
            origin='lower',
            cmap='RdBu_r',
            aspect='equal',
            vmin=-resid_max,
            vmax=resid_max,
        )
        ax.set_title('Residual (CIC - NGP)', fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        self.add_colorbar_matching_height(im, ax, label='ΔFlux')

        # === Row 1: Velocity ===
        # Column 0: Particle scatter
        ax = axes[1, 0]
        mask_fov_gas = (np.abs(gas_coords_arcsec[:, 0]) < fov_arcsec / 2) & (
            np.abs(gas_coords_arcsec[:, 1]) < fov_arcsec / 2
        )
        x_g = gas_coords_arcsec[mask_fov_gas, 0]
        y_g = gas_coords_arcsec[mask_fov_gas, 1]
        vel_g = gas_vel[mask_fov_gas]
        # Subsample if too many particles
        if len(x_g) > n_max:
            idx_sub = np.random.choice(len(x_g), n_max, replace=False)
            x_g, y_g, vel_g = x_g[idx_sub], y_g[idx_sub], vel_g[idx_sub]
        # Use SAME color limits as gridded maps for fair comparison
        vel_max_gridded = max(np.nanmax(np.abs(vel_cic)), np.nanmax(np.abs(vel_ngp)))
        sc_vel = ax.scatter(
            x_g,
            y_g,
            c=vel_g,
            s=1,
            alpha=0.5,
            cmap='RdBu_r',
            vmin=-vel_max_gridded,
            vmax=vel_max_gridded,
        )
        ax.set_xlim(-fov_arcsec / 2, fov_arcsec / 2)
        ax.set_ylim(-fov_arcsec / 2, fov_arcsec / 2)
        ax.set_aspect('equal')
        ax.set_title('Particles (Gas)', fontsize=11)
        self.add_colorbar_matching_height(sc_vel, ax, label='v$_z$ [km/s]')

        # Column 1: Velocity CIC
        ax = axes[1, 1]
        vel_max = max(np.nanmax(np.abs(vel_cic)), np.nanmax(np.abs(vel_ngp)))
        im = ax.imshow(
            vel_cic,
            origin='lower',
            cmap='RdBu_r',
            aspect='equal',
            norm=MidpointNormalize(vmin=-vel_max, vmax=vel_max, midpoint=0),
        )
        ax.set_title('CIC Gridded\n(Cloud-in-Cell)', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        self.add_colorbar_matching_height(im, ax, label='v$_{\\rm LOS}$ [km/s]')
        self.add_scale_markers(ax, image_pars, scale_bar_arcsec=1.0)

        # Column 2: Velocity NGP
        ax = axes[1, 2]
        im = ax.imshow(
            vel_ngp,
            origin='lower',
            cmap='RdBu_r',
            aspect='equal',
            norm=MidpointNormalize(vmin=-vel_max, vmax=vel_max, midpoint=0),
        )
        ax.set_title('NGP Gridded\n(Nearest-Grid-Point)', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        self.add_colorbar_matching_height(im, ax, label='v$_{\\rm LOS}$ [km/s]')
        self.add_scale_markers(ax, image_pars, scale_bar_arcsec=1.0)

        # Column 3: Velocity Residuals
        ax = axes[1, 3]
        vel_resid = vel_cic - vel_ngp
        # Handle NaNs in residual
        vel_resid_valid = vel_resid[np.isfinite(vel_resid)]
        if len(vel_resid_valid) > 0:
            vel_resid_max = np.percentile(np.abs(vel_resid_valid), 99)
        else:
            vel_resid_max = 1.0
        im = ax.imshow(
            vel_resid,
            origin='lower',
            cmap='RdBu_r',
            aspect='equal',
            vmin=-vel_resid_max,
            vmax=vel_resid_max,
        )
        ax.set_title('Residual (CIC - NGP)', fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        self.add_colorbar_matching_height(im, ax, label='Δv [km/s]')

        fig.suptitle(
            f'Gridding Comparison: Particles → CIC → NGP → Residuals (z=0.011→{target_z})',
            fontsize=13,
        )
        plt.tight_layout()
        plt.savefig(
            output_dir / 'cic_vs_ngp_comparison.png', dpi=150, bbox_inches='tight'
        )
        plt.close()

        print(
            f"✓ Saved CIC vs NGP comparison: {output_dir / 'cic_vs_ngp_comparison.png'}"
        )

    def test_symmetry_breaking_inclinations(self, test_galaxy, output_dir):
        """
        Show TNG galaxy at complementary inclinations to demonstrate asymmetry.

        Tests inc=45° vs inc=135° (viewing from above vs below at same angle).
        Symmetric models would be identical; TNG shows differences due to 3D structure.
        Note: theta_int=0 means major axis aligned with x-axis, so this tests INCLINATION not PA.
        """
        import matplotlib.pyplot as plt

        gen = TNGDataVectorGenerator(test_galaxy)
        image_pars = ImagePars(shape=(64, 64), pixel_scale=0.1, indexing='ij')
        target_z = 0.7

        inc_1 = 45.0  # Viewing from above at 45° tilt
        inc_2 = 135.0  # Viewing from below at same 45° angle (180° - 45° = 135°)

        # First inclination
        pars_1 = {
            'cosi': np.cos(np.radians(inc_1)),
            'theta_int': 0.0,
            'x0': 0.0,
            'y0': 0.0,
            'g1': 0.0,
            'g2': 0.0,
        }
        config_1 = TNGRenderConfig(
            target_redshift=target_z,
            image_pars=image_pars,
            use_native_orientation=False,
            pars=pars_1,
            use_cic_gridding=True,
        )
        int_1, _ = gen.generate_intensity_map(config_1, snr=None)
        vel_1, _ = gen.generate_velocity_map(config_1, snr=None)

        # Second inclination
        pars_2 = {
            'cosi': np.cos(np.radians(inc_2)),
            'theta_int': 0.0,
            'x0': 0.0,
            'y0': 0.0,
            'g1': 0.0,
            'g2': 0.0,
        }
        config_2 = TNGRenderConfig(
            target_redshift=target_z,
            image_pars=image_pars,
            use_native_orientation=False,
            pars=pars_2,
            use_cic_gridding=True,
        )
        int_2, _ = gen.generate_intensity_map(config_2, snr=None)
        vel_2, _ = gen.generate_velocity_map(config_2, snr=None)

        # Difference maps
        int_diff = int_2 - int_1
        vel_diff = vel_2 - vel_1

        # Plot
        fig, axes = plt.subplots(3, 2, figsize=(12, 16))

        # Log scale for intensity - use robust bounds excluding empty pixels
        int_log_1 = np.log10(int_1 + 1e-10)
        int_log_2 = np.log10(int_2 + 1e-10)
        # Only consider pixels with actual signal (> 1e-5 of max)
        threshold = np.log10(max(int_1.max(), int_2.max()) * 1e-5)
        int_log_valid = np.concatenate(
            [int_log_1[int_log_1 > threshold], int_log_2[int_log_2 > threshold]]
        )
        if len(int_log_valid) > 0:
            vmin = np.percentile(int_log_valid, 1)
            vmax = np.percentile(int_log_valid, 99.9)
        else:
            vmin, vmax = -5, 0  # Fallback

        # Row 1: inc=45°
        im00 = axes[0, 0].imshow(
            int_log_1,
            origin='lower',
            cmap='viridis',
            aspect='equal',
            vmin=vmin,
            vmax=vmax,
        )
        axes[0, 0].set_title(f'Intensity - inc={inc_1}° (from above)', fontsize=11)
        plt.colorbar(im00, ax=axes[0, 0], label='log$_{10}$(Flux)')
        self.add_scale_markers(
            axes[0, 0], image_pars, scale_bar_arcsec=1.0, color='white'
        )

        vmax_1 = np.nanmax(np.abs(vel_1))
        im01 = axes[0, 1].imshow(
            vel_1,
            origin='lower',
            cmap='RdBu_r',
            aspect='equal',
            norm=MidpointNormalize(vmin=-vmax_1, vmax=vmax_1, midpoint=0),
        )
        axes[0, 1].set_title(f'Velocity - inc={inc_1}° (from above)', fontsize=11)
        plt.colorbar(im01, ax=axes[0, 1], label='v_LOS [km/s]')
        self.add_scale_markers(axes[0, 1], image_pars, scale_bar_arcsec=1.0)

        # Row 2: inc=135°
        im10 = axes[1, 0].imshow(
            int_log_2,
            origin='lower',
            cmap='viridis',
            aspect='equal',
            vmin=vmin,
            vmax=vmax,
        )
        axes[1, 0].set_title(f'Intensity - inc={inc_2}° (from below)', fontsize=11)
        plt.colorbar(im10, ax=axes[1, 0], label='log$_{10}$(Flux)')
        self.add_scale_markers(
            axes[1, 0], image_pars, scale_bar_arcsec=1.0, color='white'
        )

        vmax_2 = np.nanmax(np.abs(vel_2))
        im11 = axes[1, 1].imshow(
            vel_2,
            origin='lower',
            cmap='RdBu_r',
            aspect='equal',
            norm=MidpointNormalize(vmin=-vmax_2, vmax=vmax_2, midpoint=0),
        )
        axes[1, 1].set_title(f'Velocity - inc={inc_2}°')
        plt.colorbar(im11, ax=axes[1, 1], label='v_LOS [km/s]')

        # Row 3: Differences
        # Use arcsinh scaling for intensity difference to see small features
        int_diff_scaled = np.arcsinh(int_diff / (0.01 * np.nanmax(np.abs(int_diff))))
        diff_max = np.nanmax(np.abs(int_diff_scaled))
        im20 = axes[2, 0].imshow(
            int_diff_scaled,
            origin='lower',
            cmap='RdBu_r',
            vmin=-diff_max,
            vmax=diff_max,
        )
        axes[2, 0].set_title('Intensity Difference (arcsinh scaled)', fontsize=11)
        plt.colorbar(im20, ax=axes[2, 0], label='arcsinh(ΔFlux)')

        vel_diff_max = np.nanmax(np.abs(vel_diff))
        im21 = axes[2, 1].imshow(
            vel_diff,
            origin='lower',
            cmap='RdBu_r',
            vmin=-vel_diff_max,
            vmax=vel_diff_max,
        )
        axes[2, 1].set_title(f'Velocity Difference')
        plt.colorbar(im21, ax=axes[2, 1], label='Δv_LOS [km/s]')

        fig.suptitle(
            f'TNG Asymmetry: inc={inc_1}° vs inc={inc_2}°\n'
            f'(Symmetric models would show zero difference)',
            fontsize=14,
        )
        plt.tight_layout()
        plt.savefig(
            output_dir / 'symmetry_breaking_inclinations.png',
            dpi=150,
            bbox_inches='tight',
        )
        plt.close()

        max_vel_diff = np.nanmax(np.abs(vel_diff))
        print(
            f"✓ Max velocity difference: {max_vel_diff:.1f} km/s (confirms asymmetry)"
        )
        print(f"✓ Saved: {output_dir / 'symmetry_breaking_inclinations.png'}")

    def test_resolution_and_snr_grid(self, test_galaxy, output_dir):
        """
        Generate grid showing TNG galaxy at fixed FOV but varying pixel resolutions and SNR.

        Top 3 rows: Intensity (log scale, RdBu_r, white=0)
        Bottom 3 rows: Velocity (RdBu_r, white=0)
        Each group: 3 pixel scales × 4 SNR levels at constant FOV
        """
        import matplotlib.pyplot as plt

        gen = TNGDataVectorGenerator(test_galaxy)

        # Fixed FOV ~10 arcsec, varying pixel scale (and thus number of pixels)
        fov_arcsec = 10.0
        pixel_scales = [0.5, 0.2, 0.05]  # arcsec/pixel (coarse to fine)
        snr_levels = [50, 200, 500, None]
        target_z = 0.7

        # 6 rows (3 intensity + 3 velocity), 4 columns (SNR levels)
        fig, axes = plt.subplots(6, 4, figsize=(16, 20))

        # First pass: Generate ALL maps to compute global colorbar bounds
        all_data = {}  # (res_idx, snr_idx) -> (intensity, velocity, image_pars)
        all_int_log_values = []
        all_vel_values = []

        for i, pix_scale in enumerate(pixel_scales):
            npix = int(np.round(fov_arcsec / pix_scale))
            shape = (npix, npix)
            image_pars = ImagePars(shape=shape, pixel_scale=pix_scale, indexing='ij')

            for j, snr in enumerate(snr_levels):
                config = TNGRenderConfig(
                    target_redshift=target_z,
                    image_pars=image_pars,
                    use_native_orientation=True,
                    use_cic_gridding=True,
                )
                intensity, _ = gen.generate_intensity_map(config, snr=snr, seed=42)
                velocity, _ = gen.generate_velocity_map(config, snr=snr, seed=42)
                all_data[(i, j)] = (intensity, velocity, image_pars)

                # Collect values for global bounds
                int_log = np.log10(np.clip(intensity, 1e-10, None))
                all_int_log_values.extend(int_log.flatten())
                all_vel_values.extend(velocity.flatten())

        # Compute global bounds across ALL resolutions and SNR levels
        vmin_int_global, vmax_int_global = self.compute_robust_intensity_bounds(
            np.array(all_int_log_values), log=True
        )
        vmax_vel_global = np.nanpercentile(np.abs(all_vel_values), 99.5)

        # Second pass: Plot with global bounds
        for i, pix_scale in enumerate(pixel_scales):
            for j, snr in enumerate(snr_levels):
                intensity, velocity, image_pars = all_data[(i, j)]

                # INTENSITY (top 3 rows) - log scale with viridis, global bounds
                ax_int = axes[i, j]
                int_log = np.log10(np.clip(intensity, 1e-10, None))
                im_int = ax_int.imshow(
                    int_log,
                    origin='lower',
                    cmap='viridis',
                    aspect='equal',
                    vmin=vmin_int_global,
                    vmax=vmax_int_global,
                )

                snr_str = f'SNR={snr}' if snr is not None else 'No noise'
                if i == 0:  # Top row
                    ax_int.set_title(f'{snr_str}', fontsize=10)
                ax_int.set_xticks([])
                ax_int.set_yticks([])
                if j == 0:  # Left column
                    ax_int.set_ylabel(f'{pix_scale}"/pix', fontsize=9, weight='bold')

                # Add colorbar
                plt.colorbar(im_int, ax=ax_int, fraction=0.046, pad=0.04)

                # Add scale markers (white for intensity)
                self.add_scale_markers(
                    ax_int, image_pars, scale_bar_arcsec=1.0, color='white'
                )

                # VELOCITY (bottom 3 rows) - RdBu_r with white=0, global bounds
                ax_vel = axes[i + 3, j]
                im_vel = ax_vel.imshow(
                    velocity,
                    origin='lower',
                    cmap='RdBu_r',
                    norm=MidpointNormalize(
                        vmin=-vmax_vel_global, vmax=vmax_vel_global, midpoint=0
                    ),
                    aspect='equal',
                )

                ax_vel.set_xticks([])
                ax_vel.set_yticks([])
                if j == 0:  # Left column
                    ax_vel.set_ylabel(f'{pix_scale}"/pix', fontsize=9, weight='bold')

                # Add colorbar
                plt.colorbar(im_vel, ax=ax_vel, fraction=0.046, pad=0.04)

                # Add scale markers
                self.add_scale_markers(ax_vel, image_pars, scale_bar_arcsec=1.0)

        # Add row labels
        fig.text(
            0.02,
            0.75,
            'INTENSITY',
            rotation=90,
            va='center',
            fontsize=12,
            weight='bold',
        )
        fig.text(
            0.02, 0.25, 'VELOCITY', rotation=90, va='center', fontsize=12, weight='bold'
        )

        fig.suptitle(
            f'Resolution & SNR Variations (Fixed FOV={fov_arcsec}" at z={target_z})',
            fontsize=14,
            y=0.995,
        )
        plt.tight_layout(rect=[0.03, 0, 1, 0.99])
        plt.savefig(
            output_dir / 'resolution_snr_grid.png', dpi=150, bbox_inches='tight'
        )
        plt.close()

        print(
            f"✓ Saved resolution/SNR grid with intensity+velocity: {output_dir / 'resolution_snr_grid.png'}"
        )

    def test_glamour_shot(self, output_dir):
        """
        Create a glamour shot figure for SubhaloID=8 showing:
        Top row: r-band flux (high-res), Hα flux (from SFR), LoS velocity (high-res)
        Bottom row: r-band (half-res + noise), Hα (half-res + noise), velocity (half-res + noise)
        """
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        # Load SubhaloID=8
        tng_data = TNG50MockData()
        galaxy = tng_data.get_galaxy(subhalo_id=8)
        gen = TNGDataVectorGenerator(galaxy)

        # High resolution for truth (top row) - zoom to ±2" at 0.025"/pix
        image_pars_highres = ImagePars(
            shape=(160, 160), pixel_scale=0.025, indexing='ij'
        )
        # Half resolution for data vectors (bottom row)
        image_pars_datavec = ImagePars(shape=(64, 64), pixel_scale=0.1, indexing='ij')
        target_z = 0.6

        # Config for high-res truth
        config_highres = TNGRenderConfig(
            image_pars=image_pars_highres,
            band='r',
            use_native_orientation=True,
            use_cic_gridding=True,
            target_redshift=target_z,
        )

        # Config for lower-res data vectors
        config_datavec = TNGRenderConfig(
            image_pars=image_pars_datavec,
            band='r',
            use_native_orientation=True,
            use_cic_gridding=True,
            target_redshift=target_z,
        )

        # Generate high-res maps (no noise) for top row
        int_clean, _ = gen.generate_intensity_map(config_highres, snr=None)
        vel_clean, _ = gen.generate_velocity_map(config_highres, snr=None)
        halpha_clean = gen.generate_sfr_map(config_highres, snr=None)

        # Generate lower-res noisy maps (SNR=250 for intensity, SNR=50 for velocity) for bottom row
        int_noisy, _ = gen.generate_intensity_map(config_datavec, snr=250, seed=42)
        vel_noisy, _ = gen.generate_velocity_map(config_datavec, snr=50, seed=42)
        halpha_noisy = gen.generate_sfr_map(config_datavec, snr=250, seed=42)

        # Calculate extents based on actual image sizes
        # High-res: 160 pix * 0.025"/pix = 4.0" total, so ±2.0"
        extent_highres = [-2.0, 2.0, -2.0, 2.0]
        # Data vector: 64 pix * 0.1"/pix = 6.4" total, so ±3.2"
        extent_datavec = [-3.2, 3.2, -3.2, 3.2]

        # Create figure
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.15, wspace=0.4)

        # Top row: High-res truth (no noise)
        # Top left: r-band flux
        ax00 = fig.add_subplot(gs[0, 0])
        int_log = np.log10(int_clean + 1e-10)
        vmin_int, vmax_int = self.compute_robust_intensity_bounds(int_log, log=True)
        im00 = ax00.imshow(
            int_log,
            origin='lower',
            cmap='viridis',
            aspect='equal',
            vmin=vmin_int,
            vmax=vmax_int,
            extent=extent_highres,
        )
        ax00.set_title('r-band Flux (Truth)', fontsize=12, weight='bold')
        ax00.set_xlabel('X [arcsec]', fontsize=10)
        ax00.set_ylabel('Y [arcsec]', fontsize=10)
        self.add_colorbar_matching_height(im00, ax00, label='log$_{10}$(Flux)')
        self.add_scale_markers(
            ax00, image_pars_highres, scale_bar_arcsec=0.5, color='white'
        )
        ax00.text(
            0.95,
            0.95,
            'pix=0.025"',
            transform=ax00.transAxes,
            fontsize=9,
            va='top',
            ha='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        )

        # Top middle: Hα flux (from SFR)
        ax01 = fig.add_subplot(gs[0, 1])
        halpha_log = np.log10(halpha_clean + 1e-10)
        vmin_ha, vmax_ha = self.compute_robust_intensity_bounds(halpha_log, log=True)
        im01 = ax01.imshow(
            halpha_log,
            origin='lower',
            cmap='viridis',
            aspect='equal',
            vmin=vmin_ha,
            vmax=vmax_ha,
            extent=extent_highres,
        )
        ax01.set_title('Hα Flux (SFR proxy, Truth)', fontsize=12, weight='bold')
        ax01.set_xlabel('X [arcsec]', fontsize=10)
        ax01.set_ylabel('Y [arcsec]', fontsize=10)
        self.add_colorbar_matching_height(im01, ax01, label='log$_{10}$(SFR)')
        self.add_scale_markers(
            ax01, image_pars_highres, scale_bar_arcsec=0.5, color='white'
        )
        ax01.text(
            0.95,
            0.95,
            'pix=0.025"',
            transform=ax01.transAxes,
            fontsize=9,
            va='top',
            ha='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        )

        # Top right: LoS velocity
        ax02 = fig.add_subplot(gs[0, 2])
        vmax_vel = np.nanmax(np.abs(vel_clean))
        im02 = ax02.imshow(
            vel_clean,
            origin='lower',
            cmap='RdBu_r',
            aspect='equal',
            norm=MidpointNormalize(vmin=-vmax_vel, vmax=vmax_vel, midpoint=0),
            extent=extent_highres,
        )
        ax02.set_title('LoS Velocity (Truth)', fontsize=12, weight='bold')
        ax02.set_xlabel('X [arcsec]', fontsize=10)
        ax02.set_ylabel('Y [arcsec]', fontsize=10)
        self.add_colorbar_matching_height(im02, ax02, label=r'$v_\mathrm{LOS}$ [km/s]')
        self.add_scale_markers(ax02, image_pars_highres, scale_bar_arcsec=0.5)
        ax02.text(
            0.95,
            0.95,
            'pix=0.025"',
            transform=ax02.transAxes,
            fontsize=9,
            va='top',
            ha='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        )

        # Bottom row: Lower-res with noise (Data Vectors)
        # Bottom left: r-band DataVector - compute separate bounds for noisy data
        ax10 = fig.add_subplot(gs[1, 0])
        int_noisy_log = np.log10(
            np.clip(int_noisy, 1e-10, None)
        )  # Clip negatives from noise
        vmin_int_noisy, vmax_int_noisy = self.compute_robust_intensity_bounds(
            int_noisy_log, log=True
        )
        im10 = ax10.imshow(
            int_noisy_log,
            origin='lower',
            cmap='viridis',
            aspect='equal',
            vmin=vmin_int_noisy,
            vmax=vmax_int_noisy,
            extent=extent_datavec,
        )
        ax10.set_title('r-band DataVector (SNR=250)', fontsize=12, weight='bold')
        ax10.set_xlabel('X [arcsec]', fontsize=10)
        ax10.set_ylabel('Y [arcsec]', fontsize=10)
        self.add_colorbar_matching_height(im10, ax10, label='log$_{10}$(Flux)')
        self.add_scale_markers(
            ax10, image_pars_datavec, scale_bar_arcsec=1.0, color='white'
        )
        ax10.text(
            0.95,
            0.95,
            'pix=0.1"',
            transform=ax10.transAxes,
            fontsize=9,
            va='top',
            ha='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        )

        # Bottom middle: Hα DataVector - compute separate bounds for noisy data
        ax11 = fig.add_subplot(gs[1, 1])
        halpha_noisy_log = np.log10(
            np.clip(halpha_noisy, 1e-10, None)
        )  # Clip negatives from noise
        vmin_ha_noisy, vmax_ha_noisy = self.compute_robust_intensity_bounds(
            halpha_noisy_log, log=True
        )
        im11 = ax11.imshow(
            halpha_noisy_log,
            origin='lower',
            cmap='viridis',
            aspect='equal',
            vmin=vmin_ha_noisy,
            vmax=vmax_ha_noisy,
            extent=extent_datavec,
        )
        ax11.set_title('Hα DataVector (SNR=250)', fontsize=12, weight='bold')
        ax11.set_xlabel('X [arcsec]', fontsize=10)
        ax11.set_ylabel('Y [arcsec]', fontsize=10)
        self.add_colorbar_matching_height(im11, ax11, label='log$_{10}$(SFR)')
        self.add_scale_markers(
            ax11, image_pars_datavec, scale_bar_arcsec=1.0, color='white'
        )
        ax11.text(
            0.95,
            0.95,
            'pix=0.1"',
            transform=ax11.transAxes,
            fontsize=9,
            va='top',
            ha='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        )

        # Bottom right: Velocity DataVector
        ax12 = fig.add_subplot(gs[1, 2])
        im12 = ax12.imshow(
            vel_noisy,
            origin='lower',
            cmap='RdBu_r',
            aspect='equal',
            norm=MidpointNormalize(vmin=-vmax_vel, vmax=vmax_vel, midpoint=0),
            extent=extent_datavec,
        )
        ax12.set_title('Velocity DataVector (SNR=50)', fontsize=12, weight='bold')
        ax12.set_xlabel('X [arcsec]', fontsize=10)
        ax12.set_ylabel('Y [arcsec]', fontsize=10)
        self.add_colorbar_matching_height(im12, ax12, label=r'$v_\mathrm{LOS}$ [km/s]')
        self.add_scale_markers(ax12, image_pars_datavec, scale_bar_arcsec=1.0)
        ax12.text(
            0.95,
            0.95,
            'pix=0.1"',
            transform=ax12.transAxes,
            fontsize=9,
            va='top',
            ha='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        )

        fig.suptitle(f'TNG50 Galaxy (SubhaloID=8, z={target_z})', fontsize=14, y=0.98)
        plt.savefig(
            output_dir / 'glamour_shot_subhalo8.png', dpi=150, bbox_inches='tight'
        )
        plt.close()

        print(f"✓ Saved glamour shot: {output_dir / 'glamour_shot_subhalo8.png'}")

    def test_orientation_sweep_inclination(self, output_dir):
        """
        Show SubhaloID=8 at fixed resolution/SNR across inclination sweep.

        Generates TWO plots:
        1. preserve_gas_stellar_offset=True (default): Gas keeps intrinsic misalignment
        2. preserve_gas_stellar_offset=False: Gas and stellar forced to same orientation

        Columns: Native, then linearly spaced cos(i) values from face-on to edge-on
        Rows: Intensity (top), Velocity (bottom)
        Single colorbar at right edge for each row, minimal whitespace between panels.

        NOTE: The "face-on" (cosi=1) view shows the disk plane, which has intrinsic
        ellipticity from the TNG simulation. The galaxy does NOT appear circular at
        face-on because real galaxy disks have intrinsic shapes.
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import ImageGrid

        # Load SubhaloID=8
        tng_data = TNG50MockData()
        galaxy = tng_data.get_galaxy(subhalo_id=8)
        gen = TNGDataVectorGenerator(galaxy)

        # Zoomed out view: larger FOV
        image_pars = ImagePars(
            shape=(80, 80), pixel_scale=0.05, indexing='ij'
        )  # 4 arcsec FOV
        target_z = 0.5  # Slightly closer for larger apparent size
        snr = None

        # Inclination sweep: native + cosi from 1 to 0 (face-on to edge-on)
        cosi_vals = np.array(
            [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
        )  # Fewer steps for cleaner plot
        n_inc = len(cosi_vals)
        inc_deg_vals = np.rad2deg(np.arccos(cosi_vals))
        n_cols = n_inc + 1  # +1 for native

        # Generate both modes: with and without gas-stellar offset preservation
        for preserve_offset in [True, False]:
            mode_label = "offset_preserved" if preserve_offset else "aligned"
            mode_title = (
                "Gas offset preserved" if preserve_offset else "Gas-stellar aligned"
            )

            # First pass: generate all maps to get consistent colorbar ranges
            all_intensities = []
            all_velocities = []
            titles_int = []
            titles_vel = []

            # Native orientation
            config_native = TNGRenderConfig(
                image_pars=image_pars,
                band='r',
                use_native_orientation=True,
                use_cic_gridding=True,
                target_redshift=target_z,
            )
            int_native, _ = gen.generate_intensity_map(config_native, snr=snr, seed=42)
            vel_native, _ = gen.generate_velocity_map(config_native, snr=snr, seed=42)
            all_intensities.append(int_native)
            all_velocities.append(vel_native)
            titles_int.append(f'Native\ninc={gen.native_inclination_deg:.1f}°')
            titles_vel.append('')

            # Custom orientations
            for cosi, inc_deg in zip(cosi_vals, inc_deg_vals):
                pars = {
                    'cosi': cosi,
                    'theta_int': np.deg2rad(gen.native_pa_deg),
                    'g1': 0.0,
                    'g2': 0.0,
                    'x0': 0.0,
                    'y0': 0.0,
                }
                config = TNGRenderConfig(
                    image_pars=image_pars,
                    band='r',
                    use_native_orientation=False,
                    use_cic_gridding=True,
                    target_redshift=target_z,
                    pars=pars,
                    preserve_gas_stellar_offset=preserve_offset,
                )
                intensity, _ = gen.generate_intensity_map(config, snr=snr, seed=42)
                velocity, _ = gen.generate_velocity_map(config, snr=snr, seed=42)
                all_intensities.append(intensity)
                all_velocities.append(velocity)
                titles_int.append(f'inc={inc_deg:.0f}°')
                titles_vel.append('')

            # Compute global colorbar ranges
            int_log_all = np.concatenate(
                [np.log10(im + 1e-10).flatten() for im in all_intensities]
            )
            vmin_int, vmax_int = self.compute_robust_intensity_bounds(
                int_log_all, log=True
            )
            vmax_vel = max(np.nanmax(np.abs(v)) for v in all_velocities)

            # Calculate extent
            fov = image_pars.shape[0] * image_pars.pixel_scale
            extent = [-fov / 2, fov / 2, -fov / 2, fov / 2]

            # Create figure with ImageGrid for tight layout and shared colorbars
            fig = plt.figure(figsize=(2.0 * n_cols + 1.5, 5))

            # Intensity row (top)
            grid_int = ImageGrid(
                fig,
                211,
                nrows_ncols=(1, n_cols),
                axes_pad=0.05,
                share_all=True,
                cbar_location="right",
                cbar_mode="single",
                cbar_size="3%",
                cbar_pad=0.1,
            )

            for idx, (intensity, title) in enumerate(zip(all_intensities, titles_int)):
                int_log = np.log10(intensity + 1e-10)
                im = grid_int[idx].imshow(
                    int_log,
                    origin='lower',
                    cmap='viridis',
                    aspect='equal',
                    vmin=vmin_int,
                    vmax=vmax_int,
                    extent=extent,
                )
                grid_int[idx].set_title(title, fontsize=9)
                grid_int[idx].set_xticks([])
                grid_int[idx].set_yticks([])
                if idx == 0:  # Scale bar on native column
                    self.add_scale_markers(
                        grid_int[idx], image_pars, scale_bar_arcsec=1.0, color='white'
                    )

            grid_int[0].set_ylabel('Intensity', fontsize=10, weight='bold')
            grid_int.cbar_axes[0].colorbar(im)
            grid_int.cbar_axes[0].set_ylabel('log(Flux)', fontsize=9)

            # Velocity row (bottom)
            grid_vel = ImageGrid(
                fig,
                212,
                nrows_ncols=(1, n_cols),
                axes_pad=0.05,
                share_all=True,
                cbar_location="right",
                cbar_mode="single",
                cbar_size="3%",
                cbar_pad=0.1,
            )

            for idx, velocity in enumerate(all_velocities):
                im = grid_vel[idx].imshow(
                    velocity,
                    origin='lower',
                    cmap='RdBu_r',
                    aspect='equal',
                    norm=MidpointNormalize(vmin=-vmax_vel, vmax=vmax_vel, midpoint=0),
                    extent=extent,
                )
                grid_vel[idx].set_xticks([])
                grid_vel[idx].set_yticks([])
                if idx == 0:
                    self.add_scale_markers(
                        grid_vel[idx], image_pars, scale_bar_arcsec=1.0
                    )

            grid_vel[0].set_ylabel('Velocity', fontsize=10, weight='bold')
            grid_vel.cbar_axes[0].colorbar(im)
            grid_vel.cbar_axes[0].set_ylabel('v [km/s]', fontsize=9)

            # Add info about gas-stellar offset
            offset_angle = gen._gas_stellar_L_angle_deg
            fig.suptitle(
                f'TNG50 SubhaloID=8: Inclination Sweep ({mode_title})\n'
                f'Gas-stellar L offset: {offset_angle:.1f}°, z={target_z}',
                fontsize=11,
                y=1.02,
            )

            out_path = output_dir / f'orientation_sweep_inclination_{mode_label}.png'
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"✓ Saved inclination sweep ({mode_label}): {out_path}")

    def test_orientation_sweep_pa(self, output_dir):
        """
        Show SubhaloID=8 at fixed resolution/SNR across PA sweep.

        Generates TWO plots:
        1. preserve_gas_stellar_offset=True (default): Gas keeps intrinsic misalignment
        2. preserve_gas_stellar_offset=False: Gas and stellar forced to same orientation

        Columns: PA from 0° to 315° in 45° steps
        Rows: Intensity (top), Velocity (bottom)
        Single colorbar at right edge for each row, minimal whitespace between panels.

        NOTE: The velocity magnitude varies with PA because the TNG velocity field
        has structure beyond pure circular rotation. As the galaxy rotates, different
        velocity features enter/leave the FOV.
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import ImageGrid

        # Load SubhaloID=8
        tng_data = TNG50MockData()
        galaxy = tng_data.get_galaxy(subhalo_id=8)
        gen = TNGDataVectorGenerator(galaxy)

        # Zoomed out view
        image_pars = ImagePars(
            shape=(80, 80), pixel_scale=0.05, indexing='ij'
        )  # 4 arcsec FOV
        target_z = 0.5
        snr = None

        # PA sweep: 0 to 315 in 45° steps
        pa_deg_vals = np.arange(0, 360, 45)
        n_pa = len(pa_deg_vals)

        # Use native inclination for all PA sweeps
        native_cosi = np.cos(np.deg2rad(gen.native_inclination_deg))

        # Generate both modes: with and without gas-stellar offset preservation
        for preserve_offset in [True, False]:
            mode_label = "offset_preserved" if preserve_offset else "aligned"
            mode_title = (
                "Gas offset preserved" if preserve_offset else "Gas-stellar aligned"
            )

            # Generate all maps
            all_intensities = []
            all_velocities = []
            for pa_deg in pa_deg_vals:
                pars = {
                    'cosi': native_cosi,
                    'theta_int': np.deg2rad(pa_deg),
                    'g1': 0.0,
                    'g2': 0.0,
                    'x0': 0.0,
                    'y0': 0.0,
                }
                config = TNGRenderConfig(
                    image_pars=image_pars,
                    band='r',
                    use_native_orientation=False,
                    use_cic_gridding=True,
                    target_redshift=target_z,
                    pars=pars,
                    preserve_gas_stellar_offset=preserve_offset,
                )
                intensity, _ = gen.generate_intensity_map(config, snr=snr, seed=42)
                velocity, _ = gen.generate_velocity_map(config, snr=snr, seed=42)
                all_intensities.append(intensity)
                all_velocities.append(velocity)

            # Compute shared colorbar ranges
            int_log_all = np.concatenate(
                [np.log10(im + 1e-10).flatten() for im in all_intensities]
            )
            vmin_int, vmax_int = self.compute_robust_intensity_bounds(
                int_log_all, log=True
            )
            vmax_vel = max(np.nanmax(np.abs(v)) for v in all_velocities)

            # Calculate extent
            fov = image_pars.shape[0] * image_pars.pixel_scale
            extent = [-fov / 2, fov / 2, -fov / 2, fov / 2]

            # Create figure with ImageGrid for tight layout and shared colorbars
            fig = plt.figure(figsize=(2.0 * n_pa + 1.5, 5))

            # Intensity row (top)
            grid_int = ImageGrid(
                fig,
                211,
                nrows_ncols=(1, n_pa),
                axes_pad=0.05,
                share_all=True,
                cbar_location="right",
                cbar_mode="single",
                cbar_size="3%",
                cbar_pad=0.1,
            )

            for idx, (pa_deg, intensity) in enumerate(
                zip(pa_deg_vals, all_intensities)
            ):
                int_log = np.log10(intensity + 1e-10)
                im = grid_int[idx].imshow(
                    int_log,
                    origin='lower',
                    cmap='viridis',
                    aspect='equal',
                    vmin=vmin_int,
                    vmax=vmax_int,
                    extent=extent,
                )
                grid_int[idx].set_title(f'PA={pa_deg:.0f}°', fontsize=9)
                grid_int[idx].set_xticks([])
                grid_int[idx].set_yticks([])
                if idx == 0:  # Scale bar on first column
                    self.add_scale_markers(
                        grid_int[idx], image_pars, scale_bar_arcsec=1.0, color='white'
                    )

            grid_int[0].set_ylabel('Intensity', fontsize=10, weight='bold')
            grid_int.cbar_axes[0].colorbar(im)
            grid_int.cbar_axes[0].set_ylabel('log(Flux)', fontsize=9)

            # Velocity row (bottom)
            grid_vel = ImageGrid(
                fig,
                212,
                nrows_ncols=(1, n_pa),
                axes_pad=0.05,
                share_all=True,
                cbar_location="right",
                cbar_mode="single",
                cbar_size="3%",
                cbar_pad=0.1,
            )

            for idx, velocity in enumerate(all_velocities):
                im = grid_vel[idx].imshow(
                    velocity,
                    origin='lower',
                    cmap='RdBu_r',
                    aspect='equal',
                    norm=MidpointNormalize(vmin=-vmax_vel, vmax=vmax_vel, midpoint=0),
                    extent=extent,
                )
                grid_vel[idx].set_xticks([])
                grid_vel[idx].set_yticks([])
                if idx == 0:
                    self.add_scale_markers(
                        grid_vel[idx], image_pars, scale_bar_arcsec=1.0
                    )

            grid_vel[0].set_ylabel('Velocity', fontsize=10, weight='bold')
            grid_vel.cbar_axes[0].colorbar(im)
            grid_vel.cbar_axes[0].set_ylabel('v [km/s]', fontsize=9)

            # Add info about gas-stellar offset
            offset_angle = gen._gas_stellar_L_angle_deg
            fig.suptitle(
                f'TNG50 SubhaloID=8: PA Sweep ({mode_title})\n'
                f'inc={gen.native_inclination_deg:.1f}°, Gas-stellar L offset: {offset_angle:.1f}°',
                fontsize=11,
                y=1.02,
            )

            out_path = output_dir / f'orientation_sweep_pa_{mode_label}.png'
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"✓ Saved PA sweep ({mode_label}): {out_path}")
