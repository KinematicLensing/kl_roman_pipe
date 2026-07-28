# COSMOS2025 downloads (COSMOS-Web master catalog)

Input catalog for the planned `cosmos25` catalog adapter: the real
COSMOS-Web master catalog (784,016 sources, 0.43 deg^2 effective;
Shuntov et al. 2025, arXiv:2506.03243) plus a private row-matched
emission-line mock painted by Jiachuan Xu. All data files are local-only
(gitignored); only this README and the download infrastructure are
committed.

Public sections (row-matched by row index across all sections):

- `COSMOSWeb_mastercatalog_<ver>_photom_primary.fits` (~2.2 GB) --
  PSF-homogenized aperture + SE++ Sersic model photometry and structure
  (`radius_sersic` is in DEGREES), flags
- `COSMOSWeb_mastercatalog_<ver>_lephare.fits` (~0.3 GB) -- LePhare
  photo-z (`zfinal`; 0 = star/artifact, filter on `type`), masses, SFRs,
  `ebv_minchi2`, object `type`
- `COSMOSWeb_mastercatalog_<ver>_cigale.fits` (~0.3 GB) -- CIGALE SED
  fits: mass, SFRs, metallicity, nonparametric SFH bins

Download (writes `<file>.provenance.json` sha256 sidecars; idempotent):

    make download-cosmos2025            # v1.1 sections
    python scripts/download_cosmos2025.py data/cosmos2025 --version v1 --files lephare

The files are world-readable, but please register once at
https://cosmos2025.iap.fr/catalog_download.html (usage tracking for the
COSMOS team). Column descriptions:
https://cosmos2025.iap.fr/catalog.html#detailed-column-descriptions

`private/` holds files that are not publicly distributable (Jiachuan
Xu's painted emission-line section and painting notebook, delivered
2026-07-28). Never commit or redistribute them.

The `cosmos25` catalog adapter consumes a single joined parquet
(`private/cosmos25_v1.parquet`, also private) built from the v1 sections
plus the painted file:

    python scripts/build_cosmos25_catalog.py    # gates the row-order join
    python scripts/audit_cosmos25_painting.py   # handshake + flux-variant audit

The painted section was built on catalog version v1 (its redshifts are
bitwise-equal to v1 `zfinal`; v1.1 revised most photo-zs), so the join
requires the v1 sections. The audit's stage-1 handshake must reproduce
the delivering notebook's densities exactly; see the adapter module
docstring (`kl_pipe/ensemble/catalogs/cosmos25.py`) for the flux-variant
definitions (dust-sign and IMF-constant corrections).

Any publication using these data must cite Shuntov et al. (2025,
arXiv:2506.03243) and Casey et al. (2023, ApJ 954, 31); see
https://cosmos2025.iap.fr/citation.html for per-product citations
(SE++ morphology, ML morphology, etc.).
