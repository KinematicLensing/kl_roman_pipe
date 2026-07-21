"""Independent GalSim-chromatic reference render of dispersed grism imaging.

Used by ``tests/test_galsim_reference.py`` to cross-check ``kl_pipe``'s
grism rendering pipeline (cube assembly + dispersion + PSF + pixel
readout) against a from-scratch construction: per-isovelocity-bin
channel images combined with ``galsim.ChromaticObject.shift`` using a
callable-of-wavelength disperser. This module tree contains no
``kl_pipe`` physics imports (only ``galsim`` + ``numpy``), so the
comparison is meaningful pixel-for-pixel.

See ``docs/validation/galsim_reference_gate.md`` for scope, frozen
tolerances, and how to run.
"""
