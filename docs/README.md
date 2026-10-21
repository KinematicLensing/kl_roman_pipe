# Documentation

## Tutorials

[`tutorials/`](tutorials/) holds Jupytext markdown tutorials (convert with `make tutorials`; see the [tutorials README](tutorials/README.md)). Suggested order: `quickstart.md` -> `intensity_models.md` -> `grism.md` -> `sampling.md`, then `roman_reference.md` and `tng50_data.md` as needed.

## Reference

- [`models.md`](models.md) - Velocity and intensity models, SourceModel key routing, observation types, adding an observation type
- [`units_and_conventions.md`](units_and_conventions.md) - Physical units and render-method input/output conventions
- [`psf_pixel_convolution.md`](psf_pixel_convolution.md) - k-space pixel integration and PSF convolution
- [`oversampling_convergence.md`](oversampling_convergence.md) - Spatial and spectral oversampling convergence
- [`fit_initialization.md`](fit_initialization.md) - Optimizer starts, MAP, Laplace metric, chain inits
- [`derivations/analytic_line_dispersal.md`](derivations/analytic_line_dispersal.md) - Analytic per-spaxel line dispersal
- [`validation/`](validation/) - GalSim reference gate, geko cross-code grism validation, rendering test coverage
- [`grism_inference_TODO.md`](grism_inference_TODO.md) - Grism inference status and deferred work
- [`ROADMAP.md`](ROADMAP.md) - Project status and planned work
- [`kl_analysis.md`](kl_analysis.md) - Modeling-complexity strategy (model misspecification vs TNG50)
- [`psf_image_shape_issue.md`](psf_image_shape_issue.md) - Resolved PSF image-shape issue (historical)

## Internal

Working records for the ensemble fitting campaigns. Not needed for single-source fitting and not kept to the same standard as the reference docs.

- [`ensemble_workflow.md`](ensemble_workflow.md) - Running `kl_pipe.ensemble` campaigns
- [`fitting_lessons.md`](fitting_lessons.md) - Lessons from ensemble sampler campaigns
- [`sampler_failure_ledger.md`](sampler_failure_ledger.md) - Per-fit sampler failure record
- [`benchmarks/`](benchmarks/) - Sampler benchmark records
- [`plans/`](plans/) - Design plans; may describe superseded APIs

## Contributing

- Tutorials go in `tutorials/` as Jupytext markdown.
- Reference material goes in this directory as plain markdown.
