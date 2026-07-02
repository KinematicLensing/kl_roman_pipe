# Tutorials

Interactive tutorials for the kinematic lensing pipeline, written as
Jupytext-compatible markdown. Read them in order, or jump to the topic you need.

## Available tutorials

- **`quickstart.md`** — the core path end to end: `SourceModel`, dotted-key
  parameters, rendering, likelihoods, optimizer recovery, the Tully-Fisher
  prior, a velocity-only NUTS fit, and the primary Roman broadband + grism
  configuration. Start here.
- **`intensity_models.md`** — the intensity model zoo, multi-component
  (bulge + disk) composites, and `RenderConfig` / rendering-accuracy control.
- **`grism.md`** — emission-line datacube assembly and grism dispersion
  (forward model only).
- **`sampling.md`** — Bayesian inference: emcee, nautilus, NumPyro NUTS, the
  Laplace preconditioner, joint photometry + grism, TNG, and diagnostics.
- **`tng50_data.md`** — rendering TNG50 mock galaxies into velocity / intensity
  data vectors. (Cells are tagged skip-execution; they need the CyVerse data.)
- **`roman_reference.md`** — a terse, full-complexity worked template for a
  realistic Roman run (multi-band coadds + multi-roll grism, WCS, realistic
  PSF). A reference, not a lesson.

## Converting to Jupyter notebooks

```bash
# Convert all tutorials
make tutorials

# Or one at a time
jupytext --to ipynb docs/tutorials/quickstart.md
```

This writes `.ipynb` files alongside the markdown, openable in Jupyter Lab /
Notebook or VS Code (select the `klpipe` kernel).

## Running and testing

```bash
make install              # create the klpipe conda env
make download-cyverse-data  # only needed for the TNG tutorials
make test-tutorials       # convert + execute the gated tutorials end to end
```

`make test-tutorials` executes quickstart, intensity_models, grism, sampling,
and tng50_data (TNG cells skip-execute without data). `roman_reference.md` is a
reference and is not in the gate; run it manually if you want to execute it.
