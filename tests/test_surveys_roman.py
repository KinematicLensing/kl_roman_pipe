"""
Literal pins and self-consistency tests for the published Roman HLWAS
survey parameters and the depth-referenced SNR machinery
(kl_pipe/surveys/roman.py + kl_pipe/noise.py compactness core).

Moved verbatim from tests/test_population.py when the survey constants
left kl_pipe/ensemble/population.py; assertion bodies and provenance
comments are unchanged apart from the module rename.
"""

import numpy as np
import pytest

from kl_pipe.surveys import roman
from kl_pipe.surveys.roman import (
    compute_line_snr_per_pass,
    compute_line_snr_total,
    matched_filter_compactness,
)


# ==============================================================================
# Matched-filter compactness + line SNR
# ==============================================================================


class TestCompactness:
    def test_monotonic_decreasing_in_reff(self):
        reff = np.linspace(0.05, 2.0, 50)
        c = matched_filter_compactness(reff, np.full(50, 0.5), np.full(50, 1.0))
        assert (np.diff(c) < 0).all()

    def test_bounds(self):
        rng = np.random.default_rng(11)
        c = matched_filter_compactness(
            rng.uniform(0.01, 3.0, 500),
            rng.uniform(0.05, 0.95, 500),
            rng.uniform(0.55, 1.9, 500),
        )
        assert (c > 0).all() and (c <= 1.0).all()

    def test_unresolved_limit(self):
        # reff << PSF -> C -> 1
        c = matched_filter_compactness(
            np.array([1e-4]), np.array([0.5]), np.array([1.0])
        )
        assert c[0] > 0.999

    def test_snr_reference_source_anchor(self):
        # the source the limit is referenced to, at exactly the per-pass
        # limit, has SNR = F_LIM_NSIGMA -- true under either reference
        snr = compute_line_snr_per_pass(
            np.array([roman.F_LIM_PER_PASS_CGS]),
            np.array([roman.fiducial_compactness()]),
        )
        np.testing.assert_allclose(snr, [roman.F_LIM_NSIGMA], rtol=1e-12)

    def test_nonpositive_reff_raises(self):
        with pytest.raises(ValueError, match='positive'):
            matched_filter_compactness(
                np.array([0.0]), np.array([0.5]), np.array([1.0])
            )


class TestLineSnrReference:
    """The source F_LIM is referenced to (roman.SNR_LINE_REFERENCE).

    Getting this wrong double-counts (or omits) the extended-source penalty.
    Wang et al. 2022 Sect. 5 derive the published Roman line-flux limits for
    a galaxy of half-light radius 0.25 arcsec at 1.5 micron, not for a point
    source; 'extended_fiducial' encodes that convention.
    """

    @pytest.fixture(autouse=True)
    def restore_reference(self):
        original = roman.SNR_LINE_REFERENCE
        yield
        roman.SNR_LINE_REFERENCE = original

    def test_default_is_extended_fiducial(self):
        # the ROTAC limit is an extended-source limit: the committee asked
        # the HLSS PIT for "line flux limits for a realistic extended source"
        # (Appendix B.2 item 1a) while asking separately for point-source and
        # extended imaging depths (B.1 item 1a), Sect. 3.1 labels only the
        # imaging depth "5 sigma point source", and Sect. 4.2.1 refers to
        # Wang et al. 2022, whose limits are for r50 = 0.25 arcsec. A
        # point-source reading would also put our yield 12x below the 14.2M
        # Halpha over 2400 deg2 that ROTAC forecasts in Sect. 4.4.3.
        assert roman.SNR_LINE_REFERENCE == 'extended_fiducial'
        assert 0.0 < roman.fiducial_compactness() < 1.0

    def test_point_source_reference_is_identity(self):
        # a point source (C = 1) at exactly the per-pass limit has
        # SNR = F_LIM_NSIGMA when the limit is referenced to a point source
        roman.SNR_LINE_REFERENCE = 'point_source'
        assert roman.fiducial_compactness() == 1.0
        snr = compute_line_snr_per_pass(
            np.array([roman.F_LIM_PER_PASS_CGS]), np.array([1.0])
        )
        assert snr[0] == pytest.approx(roman.F_LIM_NSIGMA, rel=1e-12)

    def test_fiducial_galaxy_lands_on_nsigma(self):
        # self-consistency: under the extended reference, the very galaxy the
        # limit was derived for must sit exactly at F_LIM_NSIGMA when its
        # flux equals F_LIM
        z_ref = roman.F_LIM_REF_LAMBDA_A / roman.HALPHA_REST_A - 1.0
        c_fid = matched_filter_compactness(
            np.array([roman.F_LIM_REF_R50_ARCSEC]),
            np.array([1.0]),
            np.array([z_ref]),
        )
        snr = compute_line_snr_per_pass(np.array([roman.F_LIM_PER_PASS_CGS]), c_fid)
        assert snr[0] == pytest.approx(roman.F_LIM_NSIGMA, rel=1e-12)

    def test_extended_reference_rewards_compact_sources(self):
        # a point source beats the extended fiducial by exactly 1 / C_fid
        c_ref = roman.fiducial_compactness()
        assert 0.0 < c_ref < 1.0
        snr = compute_line_snr_per_pass(
            np.array([roman.F_LIM_PER_PASS_CGS]), np.array([1.0])
        )
        assert snr[0] == pytest.approx(roman.F_LIM_NSIGMA / c_ref, rel=1e-12)

    def test_reference_choice_is_a_pure_rescaling(self):
        # switching reference must not change the RANKING of galaxies, only
        # the overall normalization -- the physics of C is untouched
        f = np.array([1e-16, 3e-16, 5e-16])
        c = np.array([0.2, 0.5, 0.9])
        c_fid = roman.fiducial_compactness()
        extended = compute_line_snr_per_pass(f, c)
        roman.SNR_LINE_REFERENCE = 'point_source'
        point = compute_line_snr_per_pass(f, c)
        ratio = extended / point
        assert np.allclose(ratio, ratio[0], rtol=1e-12)
        assert ratio[0] == pytest.approx(1.0 / c_fid)

    def test_unknown_reference_raises(self):
        roman.SNR_LINE_REFERENCE = 'not_a_reference'
        with pytest.raises(ValueError, match='SNR_LINE_REFERENCE'):
            roman.fiducial_compactness()


class TestLineSnrCoadd:
    """Per-pass vs coadded line SNR.

    The published limit is the coadd over N_GRISM_PASSES passes; a single
    pass is sqrt(N) shallower. Selection cuts belong on the total (the depth
    the joint multi-roll fit sees), while each roll's mock noise is
    normalized to one pass.
    """

    def test_total_is_sqrt_n_times_per_pass(self):
        f = np.array([1e-16, 4e-16])
        c = np.array([0.3, 0.8])
        ratio = compute_line_snr_total(f, c) / compute_line_snr_per_pass(f, c)
        expected = np.sqrt(roman.N_GRISM_PASSES)
        np.testing.assert_allclose(ratio, expected, rtol=1e-12)

    def test_per_pass_limit_is_sqrt_n_shallower_than_coadd(self):
        assert roman.F_LIM_PER_PASS_CGS == pytest.approx(
            roman.F_LIM_COADD_CGS * np.sqrt(roman.N_GRISM_PASSES), rel=1e-12
        )

    def test_reference_source_at_coadd_limit_hits_nsigma(self):
        # the source the published coadded limit was derived for, at exactly
        # that flux, sits at F_LIM_NSIGMA in the coadd
        snr = compute_line_snr_total(
            np.array([roman.F_LIM_COADD_CGS]),
            np.array([roman.fiducial_compactness()]),
        )
        assert snr[0] == pytest.approx(roman.F_LIM_NSIGMA, rel=1e-12)


class TestPublishedConstants:
    """The survey numbers the whole selection chain hangs on.

    Every other test in this module references these symbolically, so
    without a literal pin an edit to any of them would rescale the selected
    number density with nothing failing. Each value is quoted from its
    source below; changing one means the source changed.
    """

    def test_flux_limit_matches_rotac(self):
        # ROTAC Final Report 2025-04-24 v3, Sect. 3.1: HLWAS medium tier
        # "1.5 x 10^-16 erg/cm2/sec (5 sigma line flux limit, texp ~ 1500
        # sec)", the coadd over the 4 grism passes of Sect. 4.4.3
        assert roman.F_LIM_COADD_CGS == 1.5e-16
        assert roman.F_LIM_NSIGMA == 5.0
        assert roman.N_GRISM_PASSES == 4

    def test_extended_fiducial_matches_wang22(self):
        # Wang et al. 2022 (arXiv:2110.01829) Sect. 5.1: limits derived
        # "for galaxies with radius 0.25 arcsec at 1.5 micron"
        assert roman.F_LIM_REF_R50_ARCSEC == 0.25
        assert roman.F_LIM_REF_LAMBDA_A == 1.5e4
        # the resulting reference compactness, which divides every galaxy's
        # C and therefore sets the overall yield normalization (re-pinned
        # from 0.4148 when EXP_R50_OVER_RSCALE unified to the exact 1.67835)
        assert roman.fiducial_compactness() == pytest.approx(0.41486, abs=5e-5)

    def test_imaging_depths_match_rotac(self):
        # ROTAC Final Report Table 1 (arXiv:2505.10574): HLWAS medium
        # 5-sigma point-source coadded depths, F106 26.5 / F129 26.4 /
        # F158 26.4 AB ("26.4 (JH)" grouped in the table itself)
        assert roman.IMAGING_DEPTH_AB == {
            'F106': 26.5,
            'F129': 26.4,
            'F158': 26.4,
        }
        assert roman.IMAGING_DEPTH_NSIGMA == 5.0
        # AB <-> uJy pivot; f_lim(F129) = 10**((23.9 - 26.4)/2.5) = 0.1 uJy
        from kl_pipe.photometry import AB_MAG_UJY_PIVOT

        assert AB_MAG_UJY_PIVOT == 23.9
        assert roman.band_flux_limit_ujy('F129') == pytest.approx(0.1)

    def test_band_effective_wavelengths_match_galsim(self):
        # pinned literals vs galsim.roman getBandpasses effective_wavelength
        # (nm), the source recorded on BAND_EFFECTIVE_LAMBDA_A
        galsim_roman = pytest.importorskip('galsim.roman')
        bps = galsim_roman.getBandpasses()
        legacy = {'F106': 'Y106', 'F129': 'J129', 'F158': 'H158'}
        for band, lam_a in roman.BAND_EFFECTIVE_LAMBDA_A.items():
            expected_a = bps[legacy[band]].effective_wavelength * 10.0
            assert lam_a == pytest.approx(expected_a, rel=5e-4), band

    def test_extended_source_offset_reproduces_rotac_footnote(self):
        # ROTAC Table 1 caption: "r1/2 = 0.3 arcsec extended source
        # thresholds are typically ~1.1 mag brighter" than the point-source
        # depths. The Gaussian matched-filter proxy must reproduce that
        # statement; tolerance 0.3 mag covers the proxy-vs-ETC difference
        # (measured 2026-07-31: F129 1.27, F158 1.08 mag)
        for band in ('F129', 'F158'):
            c = roman.imaging_compactness(np.array([0.3]), np.array([1.0]), band)[0]
            offset_mag = -2.5 * np.log10(c)
            assert abs(offset_mag - 1.1) < 0.3, (band, offset_mag)

    def test_band_snr_and_sigma_are_consistent(self):
        # sigma_f = f / SNR by construction (the flux cancels): the two
        # helpers must agree exactly, and the flux-independence must hold
        reff = np.array([0.15, 0.3])
        cosi = np.array([0.4, 0.9])
        f = np.array([5.0, 50.0])
        snr = roman.compute_band_snr(f, reff, cosi, 'F158')
        sigma = roman.band_flux_sigma_ujy(reff, cosi, 'F158')
        np.testing.assert_allclose(f / snr, sigma, rtol=1e-12)

    def test_line_flux_sigma_is_coadd_referenced(self):
        # sigma_f = (F_LIM_COADD / 5) * C_ref / C: at the reference
        # compactness the sigma is exactly the published coadded limit / 5
        c_ref = roman.fiducial_compactness()
        sigma = roman.line_flux_sigma_cgs(np.array([c_ref]))[0]
        assert sigma == pytest.approx(roman.F_LIM_COADD_CGS / 5.0, rel=1e-12)


class TestElectronConversions:
    """Photon-count conversions for the shot-noise layer.

    Confidence anchor is Roman reference material: the conversions are
    built from the galsim.roman mission throughput tables, recomputed here
    from those tables and pinned. The 07-25 envelope arithmetic
    (~4058 e- at the published grism limit) is an independent same-inputs
    sanity check, not truth.
    """

    def test_exposure_time_pins(self):
        # ROTAC: imaging 2 passes x 3 dithers x 107.25 s; grism one pass =
        # 2 dithers x 189.75 s (8 x 189.75 = 1518 s over 4 passes)
        assert roman.T_EXP_IMAGING_S == 643.5
        assert roman.T_EXP_GRISM_PER_PASS_S == 379.5
        assert roman.T_EXP_GRISM_PER_PASS_S * roman.N_GRISM_PASSES == 1518.0

    def test_electrons_per_ujy_matches_galsim(self):
        galsim_roman = pytest.importorskip('galsim.roman')
        bps = galsim_roman.getBandpasses(AB_zeropoint=True)
        legacy = {'F106': 'Y106', 'F129': 'J129', 'F158': 'H158'}
        for band, pinned in roman.ELECTRONS_PER_UJY.items():
            zp = bps[legacy[band]].zeropoint + 2.5 * np.log10(
                galsim_roman.collecting_area
            )
            recomputed = 10 ** (-0.4 * (23.9 - zp)) * roman.T_EXP_IMAGING_S
            assert pinned == pytest.approx(recomputed, rel=1e-4), band

    def test_grism_electrons_envelope(self):
        pytest.importorskip('galsim.roman')
        # per-pass at 1.5 um (T = 0.628): pinned against the value the
        # module computed when the layer landed
        per_pass = float(roman.grism_electrons_per_f17_per_pass(1.5e4))
        assert per_pass == pytest.approx(67.64, rel=1e-3)
        # independent envelope arithmetic (07-25 handoff): a source at the
        # published coadded limit (15 in f17 units) collects ~4058 e- over
        # the 4-pass coadd
        coadd = 15.0 * roman.N_GRISM_PASSES * per_pass
        assert coadd == pytest.approx(4058.0, rel=0.01)

    def test_grism_throughput_range_raises(self):
        pytest.importorskip('galsim.roman')
        with pytest.raises(ValueError, match='outside'):
            roman.grism_throughput(9000.0)  # 900 nm, below the table
        with pytest.raises(ValueError, match='outside'):
            roman.grism_throughput(2.1e4)


class TestBackgroundAnchors:
    """Depth-anchored background sigmas (formula-level; the rendered-template
    closure lives in the roman_ensemble tier where the mock machinery is)."""

    def test_band_sigma_bg_formula(self):
        # sigma_bg = f_lim * ||K||_2 / N_sigma
        norm = 0.4
        sigma = roman.band_sigma_bg_ujy('F129', norm)
        assert sigma == pytest.approx(
            roman.band_flux_limit_ujy('F129') * norm / roman.IMAGING_DEPTH_NSIGMA,
            rel=1e-14,
        )

    def test_grism_sigma_bg_formula(self):
        # the reference template is rendered AT the per-pass limit, so
        # sigma_bg = ||L_ref||_2 / N_sigma
        assert roman.grism_sigma_bg_per_pass(10.0) == pytest.approx(2.0, rel=1e-14)

    def test_nonpositive_norms_raise(self):
        with pytest.raises(ValueError, match='psf_l2_norm'):
            roman.band_sigma_bg_ujy('F129', 0.0)
        with pytest.raises(ValueError, match='ref_line_l2_norm'):
            roman.grism_sigma_bg_per_pass(-1.0)
