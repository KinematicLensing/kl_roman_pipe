# PSFData image_shape API Inconsistency

**Status: Resolved** — PSF state now lives on `ImageObs` via `build_image_obs()`, which accepts `ImagePars` directly and handles indexing conventions internally. The old `configure_psf()` model method has been removed.

## Original Problem

The PSF module (`kl_pipe/psf.py`) used raw `(Ny, Nx)` tuples for `image_shape` parameters, bypassing the `ImagePars` abstraction that handles indexing conventions elsewhere in the pipeline. This caused test failures when `ImagePars` used `indexing='xy'`.

With `indexing='xy'`, `.shape = (Nx, Ny) = (40, 30)`, but PSFData expected `(Ny, Nx) = (30, 40)`.

Result: `ValueError: Image shape (30, 40) != PSFData.original_shape (40, 30)`

## Resolution

PSF state now lives on observation objects (`ImageObs`, `VelocityObs`), constructed via `build_image_obs()` / `build_velocity_obs()`. These accept `ImagePars` directly and handle indexing conventions internally. The old `model.configure_psf()` method has been removed.

```python
# Current API: PSF configured via observation object
obs = build_image_obs(image_pars, psf=gs_psf, data=data, variance=var)

# Pixel integration via k-space sinc (default)
obs = build_image_obs(image_pars, psf=gs_psf, data=data, variance=var)
# pixel_response=BoxPixel(pixel_scale) is the default
```

## Why ImagePars Exists

From `kl_pipe/parameters.py`:

```python
class ImagePars:
    '''
    NOTE: By default the class assumes that you are passing shape
    information in the numpy convention of (Nrow, Ncol) = (Ny, Nx)
    and that the pixel_scale is in arcsec/pixel. You can override
    this by setting indexing='xy' instead of 'ij' in the constructor.
    '''
```

The whole point of `ImagePars` is to abstract away indexing conventions. Properties like `.Nrow`, `.Ncol`, `.Ny`, `.Nx` always return the correct values regardless of how `.shape` was specified.
