"""Tests for SourceModel + EmissionLine construction and validation."""

from __future__ import annotations

import jax
import pytest

from kl_pipe.intensity import InclinedExponentialModel
from kl_pipe.source import LINE_LAMBDAS, EmissionLine, SourceModel
from kl_pipe.velocity import CenteredVelocityModel, OffsetVelocityModel


# ===========================================================================
# EmissionLine — field validation
# ===========================================================================


class TestEmissionLineValidation:

    def test_intensity_alone_ok(self):
        line = EmissionLine(intensity=InclinedExponentialModel())
        assert line.intensity is not None
        assert line.intensity_key is None

    def test_intensity_key_alone_ok(self):
        line = EmissionLine(intensity_key='Halpha')
        assert line.intensity is None
        assert line.intensity_key == 'Halpha'

    def test_neither_intensity_nor_key_raises(self):
        with pytest.raises(
            ValueError, match="exactly one of 'intensity' or 'intensity_key'"
        ):
            EmissionLine()

    def test_both_intensity_and_key_raises(self):
        with pytest.raises(
            ValueError, match="exactly one of 'intensity' or 'intensity_key'"
        ):
            EmissionLine(
                intensity=InclinedExponentialModel(),
                intensity_key='Halpha',
            )

    def test_continuum_and_continuum_key_both_raises(self):
        with pytest.raises(
            ValueError, match="at most one of 'continuum' or 'continuum_key'"
        ):
            EmissionLine(
                intensity=InclinedExponentialModel(),
                continuum=InclinedExponentialModel(),
                continuum_key='Halpha',
            )

    def test_continuum_alone_ok(self):
        line = EmissionLine(
            intensity=InclinedExponentialModel(),
            continuum=InclinedExponentialModel(),
        )
        assert line.continuum is not None
        assert line.continuum_key is None

    def test_dispersion_key_default_none(self):
        line = EmissionLine(intensity=InclinedExponentialModel())
        assert line.dispersion_key is None


# ===========================================================================
# SourceModel — construction + validation
# ===========================================================================


class TestSourceModelConstruction:

    def test_minimal_velocity_only(self):
        src = SourceModel(velocity_model=CenteredVelocityModel())
        assert src.velocity_model is not None
        assert src.broadband_models == {}
        assert src.emission_lines == {}

    def test_minimal_broadband_only(self):
        src = SourceModel(broadband_models={'F087': InclinedExponentialModel()})
        assert 'F087' in src.broadband_models

    def test_minimal_emission_only(self):
        src = SourceModel(
            emission_lines={
                'Halpha': EmissionLine(intensity=InclinedExponentialModel())
            }
        )
        assert 'Halpha' in src.emission_lines

    def test_all_three_components(self):
        src = SourceModel(
            velocity_model=OffsetVelocityModel(),
            broadband_models={'F087': InclinedExponentialModel()},
            emission_lines={
                'Halpha': EmissionLine(intensity=InclinedExponentialModel())
            },
        )
        assert src.velocity_model is not None
        assert len(src.broadband_models) == 1
        assert len(src.emission_lines) == 1


class TestSourceModelValidation:

    def test_all_empty_raises(self):
        with pytest.raises(ValueError, match="at least one of velocity_model"):
            SourceModel()

    def test_disjoint_filter_and_line_names(self):
        """Filter and line names must not collide (dotted-key disambiguation)."""
        with pytest.raises(ValueError, match="must be disjoint"):
            SourceModel(
                broadband_models={'Halpha': InclinedExponentialModel()},
                emission_lines={
                    'Halpha': EmissionLine(intensity=InclinedExponentialModel())
                },
            )


class TestLineLambdasResolution:

    def test_halpha_auto_resolved(self):
        src = SourceModel(
            emission_lines={
                'Halpha': EmissionLine(intensity=InclinedExponentialModel())
            }
        )
        assert src.emission_lines['Halpha'].lambda_rest == LINE_LAMBDAS['Halpha']
        assert src.emission_lines['Halpha'].lambda_rest == pytest.approx(656.28)

    def test_doublet_auto_resolved(self):
        src = SourceModel(
            emission_lines={
                'NII6584': EmissionLine(intensity=InclinedExponentialModel())
            }
        )
        assert src.emission_lines['NII6584'].lambda_rest == LINE_LAMBDAS['NII6584']

    def test_explicit_lambda_rest_overrides_registry(self):
        src = SourceModel(
            emission_lines={
                'Halpha': EmissionLine(
                    intensity=InclinedExponentialModel(), lambda_rest=123.45
                )
            }
        )
        assert src.emission_lines['Halpha'].lambda_rest == 123.45

    def test_unknown_line_without_lambda_rest_raises(self):
        with pytest.raises(ValueError, match="not in LINE_LAMBDAS"):
            SourceModel(
                emission_lines={
                    'MysteryLine': EmissionLine(intensity=InclinedExponentialModel())
                }
            )

    def test_unknown_line_with_explicit_lambda_rest_ok(self):
        src = SourceModel(
            emission_lines={
                'MysteryLine': EmissionLine(
                    intensity=InclinedExponentialModel(), lambda_rest=400.0
                )
            }
        )
        assert src.emission_lines['MysteryLine'].lambda_rest == 400.0


class TestIntensityKeyValidation:

    def test_intensity_key_references_existing_line(self):
        src = SourceModel(
            emission_lines={
                'Halpha': EmissionLine(intensity=InclinedExponentialModel()),
                'NII6584': EmissionLine(intensity_key='Halpha'),
            }
        )
        assert src.emission_lines['NII6584'].intensity_key == 'Halpha'

    def test_intensity_key_bad_reference_raises(self):
        with pytest.raises(ValueError, match="intensity_key='NonExistent'"):
            SourceModel(
                emission_lines={
                    'Halpha': EmissionLine(
                        intensity=InclinedExponentialModel(),
                        intensity_key=None,
                    ),
                    'NII6584': EmissionLine(intensity_key='NonExistent'),
                }
            )


class TestContinuumKeyValidation:

    def test_continuum_key_references_line_with_continuum(self):
        src = SourceModel(
            emission_lines={
                'Halpha': EmissionLine(
                    intensity=InclinedExponentialModel(),
                    continuum=InclinedExponentialModel(),
                ),
                'OIII5007': EmissionLine(
                    intensity=InclinedExponentialModel(),
                    continuum_key='Halpha',
                ),
            }
        )
        assert src.emission_lines['OIII5007'].continuum_key == 'Halpha'

    def test_continuum_key_to_line_without_continuum_raises(self):
        with pytest.raises(ValueError, match="no continuum to share"):
            SourceModel(
                emission_lines={
                    'Halpha': EmissionLine(intensity=InclinedExponentialModel()),
                    'OIII5007': EmissionLine(
                        intensity=InclinedExponentialModel(),
                        continuum_key='Halpha',
                    ),
                }
            )

    def test_continuum_key_bad_reference_raises(self):
        with pytest.raises(ValueError, match="continuum_key='NonExistent'"):
            SourceModel(
                emission_lines={
                    'Halpha': EmissionLine(
                        intensity=InclinedExponentialModel(),
                        continuum_key='NonExistent',
                    )
                }
            )


class TestDispersionKeyValidation:

    def test_dispersion_key_references_existing_line(self):
        src = SourceModel(
            emission_lines={
                'Halpha': EmissionLine(intensity=InclinedExponentialModel()),
                'NII6584': EmissionLine(
                    intensity_key='Halpha',
                    dispersion_key='Halpha',
                ),
            }
        )
        assert src.emission_lines['NII6584'].dispersion_key == 'Halpha'

    def test_dispersion_key_bad_reference_raises(self):
        with pytest.raises(ValueError, match="dispersion_key='NonExistent'"):
            SourceModel(
                emission_lines={
                    'Halpha': EmissionLine(
                        intensity=InclinedExponentialModel(),
                        dispersion_key='NonExistent',
                    )
                }
            )

    def test_chained_dispersion_key_raises(self):
        """A -> B -> C chained sharing is rejected; only one hop allowed."""
        with pytest.raises(ValueError, match="chained"):
            SourceModel(
                emission_lines={
                    'Halpha': EmissionLine(intensity=InclinedExponentialModel()),
                    'NII6584': EmissionLine(
                        intensity_key='Halpha',
                        dispersion_key='Halpha',
                    ),
                    'NII6548': EmissionLine(
                        intensity_key='Halpha',
                        dispersion_key='NII6584',
                    ),
                }
            )


class TestPytreeRegistration:

    def test_source_model_flatten(self):
        src = SourceModel(velocity_model=CenteredVelocityModel())
        leaves, treedef = jax.tree_util.tree_flatten(src)
        assert leaves == []
        # round-trip
        rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
        assert rebuilt is src

    def test_emission_line_flatten(self):
        line = EmissionLine(intensity=InclinedExponentialModel())
        leaves, treedef = jax.tree_util.tree_flatten(line)
        assert leaves == []
        rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
        assert rebuilt is line
