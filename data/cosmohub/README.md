# CosmoHub downloads (Euclid Flagship2)

Named catalog subsets for the `flagship2` catalog adapter
(`population.catalog.kind: flagship2`, the default). Each
`<name>.yaml` is a committed query spec (table, columns, region, superset
cuts); the downloaded `<name>.parquet` and its `<name>.provenance.json`
sidecar (query id, verbatim SQL, timestamps, sha256) are local-only
(gitignored; CC BY-NC 3.0 IGO license).

Download:

    make download-cosmohub-dev     # small dev/test subset (~40 MB)
    make download-cosmohub-data    # production row bank (~1 GB)

Authentication: a free CosmoHub account (https://cosmohub.pic.es) with
credentials in `~/.netrc`:

    machine api.cosmohub.pic.es
      login <email>
      password <password>

Any use of this data in publications requires the CosmoHub
acknowledgement and citations to Tallada et al. (2020) and Carretero et
al. (2018), plus the Euclid Flagship2 catalog paper (Euclid
Collaboration: Castander et al. 2024). See the CosmoHub citation guide.
