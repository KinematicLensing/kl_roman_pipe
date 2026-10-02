"""Spec knobs added for the census: escalation.ess_min_shear and dispatch.claim_order."""

import dataclasses
from pathlib import Path

import pytest
import yaml

import numpy as np
import pandas as pd

from kl_pipe.ensemble.spec import EnsembleSpec, EscalationSpec
from kl_pipe.ensemble.worker import claim_order_index, needs_escalation

pytestmark = pytest.mark.roman_ensemble

REPO_ROOT = Path(__file__).resolve().parents[1]
DEV_SPEC = REPO_ROOT / "configs" / "ensembles" / "sigma_eps_cosi_dev.yaml"


def _write(tmp_path, d):
    p = tmp_path / "spec.yaml"
    p.write_text(yaml.safe_dump(d, sort_keys=False))
    return p


class TestEssMinShear:
    def test_default_none_and_parse(self, tmp_path):
        spec = EnsembleSpec.from_yaml(DEV_SPEC)
        assert spec.escalation.ess_min_shear is None
        d = yaml.safe_load(DEV_SPEC.read_text())
        d["fit"].setdefault("escalation", {})["ess_min_shear"] = 200
        spec2 = EnsembleSpec.from_yaml(_write(tmp_path, d))
        assert spec2.escalation.ess_min_shear == 200.0
        assert spec2.resolve_defaults(d)["fit"]["escalation"]["ess_min_shear"] == 200.0
        assert (
            spec.resolve_defaults(yaml.safe_load(DEV_SPEC.read_text()))["fit"][
                "escalation"
            ]["ess_min_shear"]
            is None
        )

    def test_validation(self):
        with pytest.raises(ValueError, match="ess_min_shear"):
            EscalationSpec(ess_min_shear=0.0)
        with pytest.raises(ValueError, match="ess_min_shear"):
            EscalationSpec(ess_min_shear=-5.0)
        assert EscalationSpec(ess_min_shear=200.0).ess_min_shear == 200.0


class TestClaimOrder:
    def test_default_and_parse(self, tmp_path):
        spec = EnsembleSpec.from_yaml(DEV_SPEC)
        assert spec.claim_order == "manifest"
        d = yaml.safe_load(DEV_SPEC.read_text())
        d["dispatch"]["claim_order"] = "hard_first"
        spec2 = EnsembleSpec.from_yaml(_write(tmp_path, d))
        assert spec2.claim_order == "hard_first"
        resolved = spec2.resolve_defaults(d)["dispatch"]
        assert resolved["claim_order"] == "hard_first"
        assert resolved["workers_per_node"] == spec2.workers_per_node

    def test_validation(self, tmp_path):
        spec = EnsembleSpec.from_yaml(DEV_SPEC)
        with pytest.raises(ValueError, match="claim_order"):
            dataclasses.replace(spec, claim_order="random")
        d = yaml.safe_load(DEV_SPEC.read_text())
        d["dispatch"]["claim_orderr"] = "manifest"
        with pytest.raises(ValueError):
            EnsembleSpec.from_yaml(_write(tmp_path, d))


class TestGateAndOrder:
    def test_needs_escalation_shear_floor(self):
        healthy = {"max_rhat": 1.01, "min_ess": 150.0, "ess_g1": 260.0, "ess_g2": 120.0}
        assert not needs_escalation(healthy, EscalationSpec())
        assert needs_escalation(healthy, EscalationSpec(ess_min_shear=200.0))
        assert not needs_escalation(healthy, EscalationSpec(ess_min_shear=100.0))
        # the generic gate still applies
        assert needs_escalation(
            {**healthy, "max_rhat": 1.2}, EscalationSpec(ess_min_shear=100.0)
        )

    def test_claim_order_index(self):
        man = pd.DataFrame(
            {
                "fit_id": list("abcde"),
                "truth.cosi": [0.5, 0.06, 0.9, 0.06, 0.3],
                "line_snr": [10.0, 20.0, 10.0, 12.0, 10.0],
            }
        )
        assert list(claim_order_index(man, "manifest")) == [0, 1, 2, 3, 4]
        order = list(man.fit_id.iloc[claim_order_index(man, "hard_first")])
        # extremes first (0.06 pair, lower SNR member first), then 0.9, 0.3, 0.5
        assert order == ["d", "b", "c", "e", "a"]
        easy = list(man.fit_id.iloc[claim_order_index(man, "easy_first")])
        assert easy == order[::-1]
        with pytest.raises(ValueError, match="claim_order"):
            claim_order_index(man, "random")


class TestWarmupMetric:
    def test_default_and_parse(self, tmp_path):
        spec = EnsembleSpec.from_yaml(DEV_SPEC)
        assert spec.warmup_metric == "adapted"
        assert spec.warmup_stage2_draws == 50
        assert spec.warmup_stage2_adapt is False
        d = yaml.safe_load(DEV_SPEC.read_text())
        d["fit"].update(
            {
                "precondition": "laplace",
                "adapt_mass": True,
                "warmup_metric": "pooled",
                "warmup_stage2_draws": 100,
                "warmup_stage2_adapt": True,
            }
        )
        spec2 = EnsembleSpec.from_yaml(_write(tmp_path, d))
        assert spec2.warmup_metric == "pooled"
        assert spec2.warmup_stage2_draws == 100
        assert spec2.warmup_stage2_adapt is True
        fit = spec2.resolve_defaults(d)["fit"]
        assert fit["warmup_metric"] == "pooled"
        assert fit["warmup_stage2_draws"] == 100
        assert fit["warmup_stage2_adapt"] is True
        resolved_default = spec.resolve_defaults(yaml.safe_load(DEV_SPEC.read_text()))
        assert resolved_default["fit"]["warmup_metric"] == "adapted"

    def test_validation(self, tmp_path):
        spec = EnsembleSpec.from_yaml(DEV_SPEC)
        with pytest.raises(ValueError, match="warmup_metric"):
            dataclasses.replace(spec, warmup_metric="mean")
        with pytest.raises(ValueError, match="adapt_mass"):
            dataclasses.replace(
                spec, warmup_metric="pooled", precondition="laplace", adapt_mass=False
            )
        with pytest.raises(ValueError, match="warmup_stage2_draws"):
            dataclasses.replace(spec, warmup_stage2_draws=0)
        with pytest.raises(ValueError, match="warmup_stage2_adapt"):
            dataclasses.replace(spec, warmup_stage2_adapt="no")
        d = yaml.safe_load(DEV_SPEC.read_text())
        d["fit"]["warmup_metrics"] = "pooled"
        with pytest.raises(ValueError):
            EnsembleSpec.from_yaml(_write(tmp_path, d))


class TestPreconditionPathKnobs:
    def test_adapt_mass_and_unconstrained_require_laplace(self):
        spec = EnsembleSpec.from_yaml(DEV_SPEC)
        with pytest.raises(ValueError, match="fit.precondition: laplace"):
            dataclasses.replace(spec, precondition="none", adapt_mass=True)
        with pytest.raises(ValueError, match="fit.precondition: laplace"):
            dataclasses.replace(spec, precondition="none", unconstrained=True)
        plain = dataclasses.replace(
            spec, precondition="none", adapt_mass=False, unconstrained=False
        )
        assert plain.precondition == "none"

    def test_rollgrid_j_specs_build_sampler_config(self):
        from kl_pipe.ensemble.worker import build_sampler_config

        for name in ("rollgrid_j_isnr47", "rollgrid_j_isnr224"):
            spec = EnsembleSpec.from_yaml(DEV_SPEC.parent / f"{name}.yaml")
            assert spec.precondition == "none"
            cfg = build_sampler_config(spec)
            assert cfg.precondition == "none"
            assert cfg.n_warmup == spec.n_warmup
            # the escalation retry's overrides go through the same mapping
            cfg2 = build_sampler_config(spec, n_warmup=800, n_samples=1000)
            assert (cfg2.n_warmup, cfg2.n_samples) == (800, 1000)

    def test_expand_preflights_sampler_config(self, tmp_path, monkeypatch):
        from kl_pipe.ensemble import expander, worker

        calls = []

        def boom(spec, **kw):
            calls.append(spec.run_name)
            raise ValueError("sampler rejects this fit block")

        monkeypatch.setattr(worker, "build_sampler_config", boom)
        registry = DEV_SPEC.parent.parent / "observation"
        with pytest.raises(ValueError, match="sampler rejects"):
            expander.expand(DEV_SPEC, registry, tmp_path)
        run_name = EnsembleSpec.from_yaml(DEV_SPEC).run_name
        assert calls == [run_name]
        assert not (tmp_path / run_name).exists()


class TestRecordWarmup:
    def test_default_and_parse(self, tmp_path):
        spec = EnsembleSpec.from_yaml(DEV_SPEC)
        assert spec.record_warmup is False
        d = yaml.safe_load(DEV_SPEC.read_text())
        d["fit"].update({"precondition": "laplace", "record_warmup": True})
        spec2 = EnsembleSpec.from_yaml(_write(tmp_path, d))
        assert spec2.record_warmup is True
        assert spec2.resolve_defaults(d)["fit"]["record_warmup"] is True
        resolved_default = spec.resolve_defaults(yaml.safe_load(DEV_SPEC.read_text()))
        assert resolved_default["fit"]["record_warmup"] is False

    def test_validation(self):
        spec = dataclasses.replace(
            EnsembleSpec.from_yaml(DEV_SPEC), precondition="laplace", adapt_mass=True
        )
        with pytest.raises(ValueError, match="record_warmup"):
            dataclasses.replace(spec, record_warmup="yes")
        with pytest.raises(ValueError, match="record_warmup"):
            dataclasses.replace(spec, record_warmup=True, warmup_metric="pooled")


class TestLineSnrScale:
    CATALOG_SPEC = REPO_ROOT / "configs" / "ensembles" / "cosmos25_census_v2.yaml"

    def test_default_parse_and_resolve(self, tmp_path):
        spec = EnsembleSpec.from_yaml(self.CATALOG_SPEC)
        assert spec.line_snr_scale == 1.0
        d = yaml.safe_load(self.CATALOG_SPEC.read_text())
        d["observation"]["line_snr_scale"] = 3
        spec2 = EnsembleSpec.from_yaml(_write(tmp_path, d))
        assert spec2.line_snr_scale == 3.0
        assert spec2.resolve_defaults(d)["observation"]["line_snr_scale"] == 3.0
        assert (
            spec.resolve_defaults(yaml.safe_load(self.CATALOG_SPEC.read_text()))[
                "observation"
            ]["line_snr_scale"]
            == 1.0
        )

    def test_validation(self, tmp_path):
        d = yaml.safe_load(self.CATALOG_SPEC.read_text())
        for bad in (0, -2, True, "x"):
            d["observation"]["line_snr_scale"] = bad
            with pytest.raises(ValueError, match="line_snr_scale"):
                EnsembleSpec.from_yaml(_write(tmp_path, d))
        # sampled populations set observation.snr.line directly
        d = yaml.safe_load(DEV_SPEC.read_text())
        d["observation"]["line_snr_scale"] = 2
        with pytest.raises(ValueError, match="catalog populations only"):
            EnsembleSpec.from_yaml(_write(tmp_path, d))


class TestWallBudget:
    def test_default_none_and_parse(self, tmp_path):
        spec = EnsembleSpec.from_yaml(DEV_SPEC)
        assert spec.escalation.wall_budget_min is None
        d = yaml.safe_load(DEV_SPEC.read_text())
        d["fit"].setdefault("escalation", {})["wall_budget_min"] = 90
        spec2 = EnsembleSpec.from_yaml(_write(tmp_path, d))
        assert spec2.escalation.wall_budget_min == 90.0
        assert spec2.resolve_defaults(d)["fit"]["escalation"]["wall_budget_min"] == 90.0

    def test_validation(self):
        with pytest.raises(ValueError, match="wall_budget_min"):
            EscalationSpec(wall_budget_min=0.0)
        with pytest.raises(ValueError, match="wall_budget_min"):
            EscalationSpec(wall_budget_min=-5.0)
        assert EscalationSpec(wall_budget_min=60.0).wall_budget_min == 60.0


class TestPinToTruth:
    CATALOG_SPEC = REPO_ROOT / "configs" / "ensembles" / "cosmos25_census_v2.yaml"

    def test_default_parse_and_resolve(self, tmp_path):
        spec = EnsembleSpec.from_yaml(DEV_SPEC)
        assert spec.pin_to_truth == ()
        d = yaml.safe_load(DEV_SPEC.read_text())
        d["fit"]["pin_to_truth"] = ["x0", "y0", "vel.v0"]
        spec2 = EnsembleSpec.from_yaml(_write(tmp_path, d))
        assert spec2.pin_to_truth == ("x0", "y0", "vel.v0")
        assert spec2.resolve_defaults(d)["fit"]["pin_to_truth"] == [
            "x0",
            "y0",
            "vel.v0",
        ]
        assert (
            spec.resolve_defaults(yaml.safe_load(DEV_SPEC.read_text()))["fit"][
                "pin_to_truth"
            ]
            == []
        )

    def test_validation(self, tmp_path):
        d = yaml.safe_load(DEV_SPEC.read_text())
        d["fit"]["pin_to_truth"] = "x0"
        with pytest.raises(ValueError, match="pin_to_truth"):
            EnsembleSpec.from_yaml(_write(tmp_path, d))
        d["fit"]["pin_to_truth"] = ["x0", "x0"]
        with pytest.raises(ValueError, match="duplicates"):
            EnsembleSpec.from_yaml(_write(tmp_path, d))
        spec = EnsembleSpec.from_yaml(DEV_SPEC)
        with pytest.raises(ValueError, match="pin_to_truth"):
            dataclasses.replace(spec, pin_to_truth=("x0", ""))


class TestPaintScaleRatios:
    CATALOG_SPEC = REPO_ROOT / "configs" / "ensembles" / "cosmos25_census_v2.yaml"

    def test_default_none_and_parse(self, tmp_path):
        spec = EnsembleSpec.from_yaml(self.CATALOG_SPEC)
        cp = spec.catalog_population
        assert cp.paint_vel_rscale_ratio is None
        assert cp.paint_halpha_rscale_ratio is None
        d = yaml.safe_load(self.CATALOG_SPEC.read_text())
        d["population"]["paint"]["vel_rscale_ratio"] = {
            "median": 0.28,
            "scatter_dex": 0.3,
        }
        d["population"]["paint"]["halpha_rscale_ratio"] = {
            "median": 1.0,
            "scatter_dex": 0.2,
        }
        cp2 = EnsembleSpec.from_yaml(_write(tmp_path, d)).catalog_population
        assert cp2.paint_vel_rscale_ratio == (0.28, 0.3)
        assert cp2.paint_halpha_rscale_ratio == (1.0, 0.2)

    def test_validation(self, tmp_path):
        d = yaml.safe_load(self.CATALOG_SPEC.read_text())
        d["population"]["paint"]["vel_rscale_ratio"] = {"median": 0.28}
        with pytest.raises(ValueError, match="vel_rscale_ratio"):
            EnsembleSpec.from_yaml(_write(tmp_path, d))
        d["population"]["paint"]["vel_rscale_ratio"] = {
            "median": 0.28,
            "scatter_dex": 0.0,
        }
        with pytest.raises(ValueError, match="must both be positive"):
            EnsembleSpec.from_yaml(_write(tmp_path, d))
        d["population"]["paint"]["vel_rscale_ratio"] = {
            "median": 0.28,
            "scatter_dex": 0.3,
            "x": 1,
        }
        with pytest.raises(ValueError, match="vel_rscale_ratio"):
            EnsembleSpec.from_yaml(_write(tmp_path, d))


ROLLGRID_SPEC = REPO_ROOT / "configs" / "ensembles" / "rollgrid_j_isnr47.yaml"
ROLLGRID_HONEST_SPEC = (
    REPO_ROOT / "configs" / "ensembles" / "rollgrid_honest_isnr47.yaml"
)
F158_CONFIG = (
    REPO_ROOT / "configs" / "observation" / "hlwas_medium_roman_rotac_rolls_f158.yaml"
)


class TestPriorOverrides:
    def test_default_parse_and_resolve(self, tmp_path):
        spec = EnsembleSpec.from_yaml(DEV_SPEC)
        assert spec.prior_overrides == {}
        d = yaml.safe_load(DEV_SPEC.read_text())
        d["fit"]["prior_overrides"] = {
            "vel.rscale": {"dist": "uniform", "low": 0.001, "high": 1.0},
            "vcirc": {
                "dist": "lognormal",
                "median": 251.5,
                "sigma_dex": 0.0749,
                "clip_sigmas": 3,
            },
            "Halpha.flux": {"dist": "uniform_relative", "low": 0.1, "high": 10.0},
            "vel.v0": {"dist": "gaussian", "loc": 0.0, "scale": 125.0},
        }
        spec2 = EnsembleSpec.from_yaml(_write(tmp_path, d))
        # the vcirc alias resolves to the dotted name
        assert set(spec2.prior_overrides) == {
            "vel.rscale",
            "vel.vcirc",
            "Halpha.flux",
            "vel.v0",
        }
        assert spec2.prior_overrides["vel.vcirc"].params["clip_sigmas"] == 3.0
        resolved = spec2.resolve_defaults(d)["fit"]["prior_overrides"]
        assert resolved["vel.rscale"] == {"dist": "uniform", "low": 0.001, "high": 1.0}
        assert (
            spec.resolve_defaults(yaml.safe_load(DEV_SPEC.read_text()))["fit"][
                "prior_overrides"
            ]
            == {}
        )

    def test_validation(self, tmp_path):
        d = yaml.safe_load(DEV_SPEC.read_text())
        d["fit"]["prior_overrides"] = {
            "vel.rscale": {"dist": "cauchy", "low": 0, "high": 1}
        }
        with pytest.raises(ValueError, match="unknown prior override dist"):
            EnsembleSpec.from_yaml(_write(tmp_path, d))
        d["fit"]["prior_overrides"] = {
            "vel.rscale": {"dist": "uniform", "low": 1.0, "high": 0.5}
        }
        with pytest.raises(ValueError, match="must be > low"):
            EnsembleSpec.from_yaml(_write(tmp_path, d))
        d["fit"]["prior_overrides"] = {
            "vel.rscale": {"dist": "uniform_relative", "low": 0.0, "high": 2.0}
        }
        with pytest.raises(ValueError, match="must be > 0"):
            EnsembleSpec.from_yaml(_write(tmp_path, d))
        d["fit"]["prior_overrides"] = {"vel.rscale": {"low": 0.1, "high": 1.0}}
        with pytest.raises(ValueError, match="'dist' key"):
            EnsembleSpec.from_yaml(_write(tmp_path, d))
        d["fit"]["prior_overrides"] = {
            "vel.v0": {"dist": "gaussian", "loc": 0.0, "scale": 10.0}
        }
        d["fit"]["pin_to_truth"] = ["vel.v0"]
        with pytest.raises(ValueError, match="conflicts with fit.pin_to_truth"):
            EnsembleSpec.from_yaml(_write(tmp_path, d))

    def test_scene_priors_apply_overrides(self):
        """The forecast-match arm: 11 sampled parameters with the table's priors."""
        import math

        from kl_pipe.ensemble.expander import build_manifest
        from kl_pipe.ensemble.scene import scene_priors
        from kl_pipe.ensemble.spec import ObservationConfig
        from kl_pipe.priors import Gaussian, TruncatedLogNormal, Uniform

        spec = EnsembleSpec.from_yaml(ROLLGRID_SPEC)
        config = ObservationConfig.from_yaml(F158_CONFIG)
        m = build_manifest(spec, config)
        row = m.iloc[7]
        truth = {
            k[len("truth.") :]: float(v)
            for k, v in row.items()
            if k.startswith("truth.")
        }
        priors = scene_priors(truth, config, spec, row=row)
        assert len(priors.sampled_names) == 11
        assert set(priors.sampled_names) == {
            "F158.flux",
            "F158.rscale",
            "Halpha.flux",
            "Halpha.rscale",
            "Halpha.dispersion",
            "cosi",
            "theta_int",
            "g1",
            "g2",
            "vel.vcirc",
            "vel.rscale",
        }
        flux = priors.get_prior("F158.flux")
        assert isinstance(flux, Uniform)
        assert flux.bounds == pytest.approx(
            (0.1 * truth["F158.flux"], 10.0 * truth["F158.flux"])
        )
        tf = priors.get_prior("vel.vcirc")
        assert isinstance(tf, TruncatedLogNormal)
        sigma = 0.0749 * math.log(10.0)
        assert tf.low == pytest.approx(251.5 * math.exp(-3 * sigma))
        assert tf.high == pytest.approx(251.5 * math.exp(3 * sigma))
        assert priors.get_prior("cosi").bounds == (0.001, 0.999)
        assert isinstance(priors.get_prior("g1"), Gaussian)
        assert priors.get_prior("g1").sigma == 0.3
        # overriding a parameter the scene fixes is an error
        bad = dataclasses.replace(
            spec,
            prior_overrides={
                **spec.prior_overrides,
                "F158.h_over_r": spec.prior_overrides["vel.rscale"],
            },
        )
        with pytest.raises(ValueError, match="not a sampled scene parameter"):
            scene_priors(truth, config, bad, row=row)

    def test_honest_arm_shared_thickness(self):
        """A drawn h_over_r is one shared sampled parameter with the draw's prior."""
        from kl_pipe.ensemble.expander import build_manifest
        from kl_pipe.ensemble.scene import scene_priors
        from kl_pipe.ensemble.spec import ObservationConfig
        from kl_pipe.priors import LogNormal

        spec = EnsembleSpec.from_yaml(ROLLGRID_HONEST_SPEC)
        config = ObservationConfig.from_yaml(F158_CONFIG)
        m = build_manifest(spec, config)
        h_cols = [c for c in m.columns if c.endswith("h_over_r")]
        assert "truth.h_over_r" in h_cols and len(h_cols) == 4
        for c in h_cols:
            np.testing.assert_array_equal(
                m[c].to_numpy(), m["truth.h_over_r"].to_numpy()
            )
        assert m["truth.h_over_r"].nunique() > 1
        row = m.iloc[3]
        truth = {
            k[len("truth.") :]: float(v)
            for k, v in row.items()
            if k.startswith("truth.")
        }
        priors = scene_priors(truth, config, spec, row=row)
        assert "h_over_r" in priors.sampled_names
        assert not any(n.endswith(".h_over_r") for n in priors.sampled_names)
        assert isinstance(priors.get_prior("h_over_r"), LogNormal)
        assert "vel.v0" in priors.sampled_names
        assert len(priors.sampled_names) == 13


class TestGridAndLognormalDraws:
    def test_parse_and_validation(self, tmp_path):
        d = yaml.safe_load(DEV_SPEC.read_text())
        d["population"]["draw"]["theta_int"] = {"dist": "grid", "values": [0.0, 1.0]}
        d["population"]["draw"]["vcirc"] = {
            "dist": "lognormal_tf",
            "center_kms": 251.5,
            "sigma_tf_dex": 0.075,
            "truth_sigma_tf_dex": 0.0,
        }
        spec = EnsembleSpec.from_yaml(_write(tmp_path, d))
        assert spec.draw["theta_int"].dist == "grid"
        assert spec.draw["vel.vcirc"].params["truth_sigma_tf_dex"] == 0.0
        d["population"]["draw"]["theta_int"] = {"dist": "grid", "values": []}
        with pytest.raises(ValueError, match="non-empty"):
            EnsembleSpec.from_yaml(_write(tmp_path, d))
        d["population"]["draw"]["theta_int"] = {"dist": "grid", "values": [1.0, 1.0]}
        with pytest.raises(ValueError, match="unique"):
            EnsembleSpec.from_yaml(_write(tmp_path, d))
        d["population"]["draw"]["theta_int"] = {
            "dist": "uniform",
            "low": 0.0,
            "high": 3.0,
        }
        d["population"]["draw"]["vcirc"]["truth_sigma_tf_dex"] = -0.1
        with pytest.raises(ValueError, match="truth_sigma_tf_dex"):
            EnsembleSpec.from_yaml(_write(tmp_path, d))
        d["population"]["draw"]["vcirc"] = {
            "dist": "lognormal_tf",
            "center_kms": 200.0,
            "sigma_tf_dex": 0.08,
        }
        d["population"]["draw"]["h_over_r"] = {
            "dist": "lognormal",
            "median": 0.25,
            "sigma_dex": 0.0,
        }
        with pytest.raises(ValueError, match="lognormal draw"):
            EnsembleSpec.from_yaml(_write(tmp_path, d))

    def test_rollgrid_manifest(self):
        """Grid draws cycle by galaxy index, fixed-truth vcirc, flux-anchored labels."""
        from kl_pipe.ensemble.expander import build_manifest
        from kl_pipe.ensemble.spec import ObservationConfig
        from kl_pipe.photometry import EXP_R50_OVER_RSCALE
        from kl_pipe.surveys.roman import (
            compute_line_snr_per_pass,
            matched_filter_compactness,
        )

        spec = EnsembleSpec.from_yaml(ROLLGRID_SPEC)
        assert spec.stratify_param == "line_flux_cgs"
        config = ObservationConfig.from_yaml(F158_CONFIG)
        m = build_manifest(spec, config)
        assert len(m) == 5 * 5 * 4 == spec.n_fits
        # cos i cycles through the five grid values by galaxy index
        by_gal = m.drop_duplicates("galaxy_id").sort_values("galaxy_id")
        np.testing.assert_allclose(by_gal["truth.cosi"], [0.1, 0.3, 0.5, 0.7, 0.9])
        np.testing.assert_allclose(m["truth.theta_int"], np.pi / 2)
        assert (m["truth.vel.vcirc"] == 251.5).all()
        assert (m["truth.z"] == 1.0).all()
        assert sorted(m["line_flux_cgs"].unique()) == sorted(spec.sweep_values)
        # the label is the published-depth per-pass line SNR of that flux for
        # a galaxy of this disk r50 at its cos i and z
        for _, r in m.drop_duplicates(["sweep_step", "galaxy_id"]).iterrows():
            r50 = EXP_R50_OVER_RSCALE * r["truth.F158.rscale"]
            c = matched_filter_compactness(
                np.array([r50]), np.array([r["truth.cosi"]]), np.array([r["truth.z"]])
            )
            expect = compute_line_snr_per_pass(np.array([r["line_flux_cgs"]]), c)[0]
            assert r["line_snr"] == pytest.approx(expect, rel=1e-12)
        # more face-on, less compact, lower label at fixed flux
        step = (
            m[m["sweep_step"] == 1]
            .drop_duplicates("galaxy_id")
            .sort_values("truth.cosi")
        )
        assert (np.diff(step["line_snr"].to_numpy()) < 0).all()
        # common random numbers across the flux sweep
        for _, g in m.groupby(["galaxy_id", "noise_rep"]):
            assert g["noise_seed"].nunique() == 1
            assert len(g) == 5
