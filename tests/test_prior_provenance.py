"""
Coverage and drift guards for the prior-provenance registry.

The registry must describe exactly the parameter set the production scene
produces: a prior added without a provenance entry, or an entry left behind
after a prior is removed, fails here rather than shipping a blank or stale
table row.
"""

from pathlib import Path

import pytest
import yaml

from kl_pipe.ensemble.expander import expand, load_run, truth_from_row
from kl_pipe.ensemble.prior_provenance import (
    catalog_registry,
    registry_to_latex,
    standalone_document,
)
from kl_pipe.ensemble.scene import scene_priors

from test_population import (
    catalog_spec_dict,
    fake_flagship2_rows,
    write_fake_catalog,
)

pytestmark = pytest.mark.roman_ensemble

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY = REPO_ROOT / 'configs' / 'observation'


@pytest.fixture(scope='module', params=[True, False], ids=['sample_n', 'pin_n'])
def provenance_parts(request, tmp_path_factory):
    """Expanded small catalog run + its registry, both bulge-n conventions."""
    data_dir = tmp_path_factory.mktemp('cosmohub_fake_prov')
    write_fake_catalog(data_dir, fake_flagship2_rows(n=400, seed=99))
    d = catalog_spec_dict(data_dir)
    d['population']['sample']['n_galaxies'] = 2
    d['run']['name'] = f'prov_test_{request.param}'
    d.setdefault('fit', {})['sample_bulge_nsersic'] = request.param
    tmp = tmp_path_factory.mktemp('prov_run')
    spec_path = tmp / 'spec.yaml'
    spec_path.write_text(yaml.safe_dump(d))
    run_dir = expand(spec_path, REGISTRY, tmp / 'runs')
    spec, config, manifest = load_run(run_dir)
    row = manifest.iloc[0]
    truth = truth_from_row(row)
    priors = scene_priors(truth, config, spec, row=row)
    return spec, config, priors, catalog_registry(spec, config)


class TestRegistryCoverage:
    def test_registry_covers_exactly_the_scene_parameters(self, provenance_parts):
        spec, config, priors, registry = provenance_parts
        scene_params = set(priors.sampled_names) | set(priors.fixed_names)
        if not spec.sample_bulge_nsersic:
            # constructor-pinned, bypasses PriorDict; the registry must still
            # carry it so the leak is documented rather than invisible
            scene_params |= {f'{b}.bulge_n_sersic' for b in config.bands}
        assert set(registry) == scene_params

    def test_every_entry_complete(self, provenance_parts):
        _, _, _, registry = provenance_parts
        for param, e in registry.items():
            assert e.param == param
            for short_field in (e.meaning, e.unit, e.painted, e.fit_prior):
                assert isinstance(short_field, str) and short_field, param
            assert isinstance(e.notes, str) and len(e.notes) > 10, param

    def test_pinned_bulge_n_is_flagged_as_leak(self, provenance_parts):
        spec, config, _, registry = provenance_parts
        for band in config.bands:
            entry = registry[f'{band}.bulge_n_sersic']
            if spec.sample_bulge_nsersic:
                assert entry.category == 'paint'
            else:
                assert entry.category == 'interim'
                assert 'leak' in entry.fit_prior.lower()

    def test_latex_renders_both_modes(self, provenance_parts):
        _, _, _, registry = provenance_parts
        standalone = registry_to_latex(registry, mode='standalone')
        paper = registry_to_latex(registry, mode='paper')
        assert standalone.count(r'\\') >= len(registry)
        assert r'\citet{Ubler2017}' in paper
        # raw underscores outside \texttt break the paper build
        doc = standalone_document(registry)
        assert doc.startswith('\\documentclass')
        assert doc.rstrip().endswith(r'\end{document}')
