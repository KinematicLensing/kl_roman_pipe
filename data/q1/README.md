# Euclid Q1 validation anchor

Real-data validation sample for the Flagship2-backed ensemble: all Q1
rank-0 Halpha detections with flux > 2e-16 erg/s/cm2 and line SNR >= 5,
joined to MER VIS Sersic morphology and SPE redshifts. Retrieved from
IRSA TAP (public, no auth) by `scripts/download_q1.py`; the CSV and its
provenance sidecar are local-only (gitignored).

    make download-q1-data
