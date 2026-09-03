"""
Physical constants for the pipeline, in the pipeline's working units.

Single source of truth so a value is never hardcoded in more than one place.
Derived from astropy (authoritative CODATA / SI) rather than literals.
"""

import numpy as np
from astropy.constants import c as _c

# speed of light in km/s (c is exactly 299792458 m/s under the 2019 SI)
C_KMS = float(_c.to('km/s').value)

# radians -> arcsec (exact)
ARCSEC_PER_RAD = 180.0 / np.pi * 3600.0
