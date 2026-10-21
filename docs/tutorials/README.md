# Tutorials

Interactive tutorials for the kinematic lensing pipeline, written as
Jupytext-compatible markdown. Read them in order, or jump to the topic you need.

## Recommended order

1. **`quickstart.md`, Examples 1-5** -- the core workflow: `SourceModel`,
   dotted-key parameters, observations, noise and SNR conventions, the
   likelihood, optimizer recovery, the Tully-Fisher prior, and a velocity-only
   NUTS fit. Prerequisites: none. Examples 6-9 (Roman broadband + grism, bulge +
   disk, render grids, a multi-band multi-roll mock) are optional.
2. **`intensity_models.md`** -- the intensity model zoo, multi-component
   (bulge + disk) composites, and `RenderConfig` / rendering-accuracy control.
   Prerequisites: quickstart Examples 1-2.
3. **`grism.md`** -- emission-line datacube assembly and grism dispersion
   (forward model only). Prerequisites: quickstart Examples 1-2.
4. **`sampling.md`** -- inference in depth: emcee, nautilus, NumPyro NUTS, the
   Laplace preconditioner, fit initialization, continuing runs, joint
   photometry + grism, and diagnostics. Prerequisites: quickstart Examples 1-5;
   `grism.md` for Section 6.

As needed:

- **`roman_reference.md`** -- a terse, full-complexity template for a realistic
  Roman run (multi-band coadds, multi-roll grism, WCS, the Roman WFI PSF) and
  physical-unit noise from published survey depths. A reference, not a lesson.
  Prerequisites: quickstart Examples 1-6.
- **`tng50_data.md`** -- rendering TNG50 mock galaxies into velocity /
  intensity data vectors. Prerequisites: quickstart Examples 1-2 and the TNG50
  data (`make download-cyverse-data`), or `KLPIPE_TNG_DATA_DIR` pointing elsewhere.

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
and tng50_data (on a generated synthetic TNG fixture when `data/tng50/` is
incomplete). `roman_reference.md` is a
reference and is not in the gate; run it manually if you want to execute it.
