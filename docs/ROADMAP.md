# Roadmap

Evolving feature list for `kl_pipe`. Updated as priorities shift.

## Done

- PSF convolution with k-space pixel integration (sinc + wrap)
- 3D inclined exponential model (sech² vertical profile, k-space FFT)
- Spergel, de Vaucouleurs, Sersic (emulator), and bulge+disk composite intensity models
- MCMC sampling infrastructure (emcee, nautilus, numpyro, blackjax)
- TNG + PSF integration (`test_psf_tng.py`)
- Mask support in likelihoods (for missing/bad pixels)
- `SourceModel`: one object holding the velocity model, per-band intensity components, and emission lines (dotted parameter keys)
- Roman grism forward model: datacube assembly + dispersion at arbitrary angle (JIT + autodiff)
- Grism inference: likelihood, `InferenceTask.from_obs`, slice / optimizer / sampling tests
- Joint multi-band imaging + grism fitting, multiple grism roll angles
- Optional Poisson shot noise in synthetic imaging (`noise.add_intensity_noise`)

## Medium-term

- Chromatic PSF: per-line / wavelength-dependent grism PSFs (one shared PSF per `GrismObs` today; issue #51)
- Instrumental line-spread function
- Spatially-varying PSF (across Roman focal plane)
- Interpixel capacitance (IPC) pixel response

## Long-term

- Ensemble fitting at survey scale (internal tooling in `kl_pipe/ensemble/`, not yet supported for general use)
