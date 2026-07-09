# Documentation

This directory contains documentation for the kinematic lensing pipeline project.

## Structure

- **`tutorials/`** - Interactive tutorials for learning how to use the pipeline
  - See the [tutorials README](tutorials/README.md) for how to convert markdown files to Jupyter notebooks

## Contents

### Tutorials

The tutorials provide hands-on examples of using the pipeline components:
- **quickstart.md** - Introduction to the basic pipeline functionality
- **intensity_models.md** - The intensity model zoo (exponential, Spergel, Sersic, composite)
- **grism.md** - Slitless grism dispersion and joint photometry+grism inference
- **sampling.md** - MCMC sampling with NumPyro (Laplace preconditioning, diagnostics)
- **roman_reference.md** - Roman-realistic WCS + PSF reference setup
- **tng50_data.md** - Working with TNG50 mock observations from CyVerse

All tutorials are written in Jupytext-compatible markdown format and can be converted to executable Jupyter notebooks.

## Contributing

When adding new documentation:
- Place tutorials in the `tutorials/` directory
- Use Jupytext-compatible markdown format for interactive content
- Keep plain markdown documentation in this directory for reference material
