# CyVerse Data Download Configuration

This directory contains configuration for downloading large test data files from CyVerse.

## Quick Start

1. **Download TNG50 mock data:** `make download-cyverse-data`
2. **Run TNG50 tests:** `make test-tng50`

To use the data in your code:
```python
from kl_pipe.tng import TNG50MockData

# Load all TNG50 mock data
mock_data = TNG50MockData()
gas = mock_data.gas
stellar = mock_data.stellar
subhalo = mock_data.subhalo

# Or load individual datasets
from kl_pipe.tng import load_gas_data, load_stellar_data
gas = load_gas_data()
stellar = load_stellar_data()
```

To add more files, edit `cyverse_data.conf` with additional URLs.

## Configuration Format

Edit `cyverse_data.conf` with one file per line:
```
REMOTE_URL | LOCAL_PATH
```

- **REMOTE_URL**: Full CyVerse WebDAV URL
- **LOCAL_PATH**: Where to save relative to `data/` directory

### Example

```
# Public data (no authentication needed)
https://data.cyverse.org/dav-anon/iplant/home/shared/kl_project/tng50/snapshot.hdf5 | tng50/snapshot.hdf5

# Private data (will prompt for credentials on first download)
https://data.cyverse.org/dav/iplant/home/myuser/private_data.fits | tng50/private_data.fits
```

## Authentication

**For public data** (URLs with `dav-anon`): No authentication needed.

**For private data**: The script will automatically prompt you to set up credentials on first download. Your credentials will be saved securely in `~/.netrc` (standard Unix credential storage).

### CyVerse URL Patterns

```bash
# Public/shared data
https://data.cyverse.org/dav-anon/iplant/home/shared/<project>/<path>

# Your personal data  
https://data.cyverse.org/dav/iplant/home/<username>/<path>

# Curated datasets (DOI-backed)
https://data.cyverse.org/dav-anon/iplant/commons/cyverse_curated/<path>
```

## Makefile Targets

```bash
make download-cyverse-data  # Download configured files
make test-tng50            # Run TNG50 tests (downloads data first if needed)
make test-all              # Run all tests including TNG50
make clean-cyverse-data    # Remove downloaded files
```

## Testing Integration

Mark tests that need CyVerse data with `@pytest.mark.tng50`:

```python
import pytest

@pytest.mark.tng50
def test_with_tng50_data():
    # Test using downloaded TNG50 data
    pass
```

This allows:
- `make test` - Basic tests only (no download required)
- `make test-tng50` - TNG50 tests (downloads data automatically)
- `make test-all` - Everything

## Notes

- Downloaded files are gitignored (not tracked)
- Re-running download skips already-downloaded files
- Some downloaded CyVerse files are large; it will take time!
