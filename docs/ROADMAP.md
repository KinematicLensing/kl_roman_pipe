# Roadmap

Evolving feature list for `kl_pipe`. Updated as priorities shift.

## Done

- PSF convolution with k-space pixel integration (sinc + wrap)
- 3D inclined exponential model (sech² vertical profile, k-space FFT)
- MCMC sampling infrastructure (emcee, nautilus, numpyro, blackjax)
- TNG + PSF integration (`test_psf_tng.py`)
- Mask support in likelihoods (for missing/bad pixels)

## In progress

- Roman grism forward model — V1 datacube + dispersion (`se/grism-core`)
  - SpectralModel: 3D datacube from velocity + intensity + emission lines
  - GrismPars + disperse_cube: sub-pixel dispersion at arbitrary angle
  - KLModel: 3-way parameter merge, render_cube, render_grism (JIT + autodiff)
  - Deferred: likelihood, InferenceTask, inference tests (see `docs/grism_inference_TODO.md`)

## Medium-term

- Grism inference layer (likelihood, InferenceTask, sampling diagnostics)
- Joint photometry + grism fitting (breaks PA-velocity degeneracy)
- Chromatic PSF support (wavelength-dependent PSF across grism bandpass)
- Spatially-varying PSF (across Roman focal plane)
- Pixel response function / IPC modeling
- Poisson noise model (beyond Gaussian)

## Long-term

- Multi-band joint fitting
- GPU-accelerated sampling
