---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.0
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Working with TNG50 Mock Data

This tutorial demonstrates how to download and work with TNG50 mock observations for the kinematic lensing pipeline.

## Prerequisites

Before running this tutorial, you need to download the TNG50 mock data from CyVerse:

```bash
# make download-cyverse-data
```

This will download three data files (~340 MB total):
- `gas_data_analysis.npz` (62 MB)
- `stellar_data_analysis.npz` (230 MB)
- `subhalo_data_analysis.npz` (48 MB)

## Loading TNG50 Data

The `kl_pipe.tng` module provides convenient functions for loading the mock data.

```python
from kl_pipe.tng import TNG50MockData, load_gas_data, load_stellar_data, load_subhalo_data
import numpy as np
import matplotlib.pyplot as plt
```

### Loading All Data at Once

The easiest way to access all TNG50 mock data is through the `TNG50MockData` class:

```python
# Load all three datasets
mock_data = TNG50MockData()

print(mock_data)
print(f"\nGas data keys: {list(mock_data.gas.keys())}")
print(f"Stellar data keys: {list(mock_data.stellar.keys())}")
print(f"Subhalo data keys: {list(mock_data.subhalo.keys())}")
```

### Loading Individual Datasets

You can also load datasets individually if you only need specific data:

```python
# Load only gas data
gas = load_gas_data()
print(f"Gas data contains: {list(gas.keys())}")
```

```python
# Load only stellar data
stellar = load_stellar_data()
print(f"Stellar data contains: {list(stellar.keys())}")
```

```python
# Load only subhalo data
subhalo = load_subhalo_data()
print(f"Subhalo data contains: {list(subhalo.keys())}")
```

### Selective Loading

For memory efficiency, you can selectively load only the datasets you need:

```python
# Load only gas and stellar data, skip subhalo
mock_data_partial = TNG50MockData(
    load_gas=True,
    load_stellar=True,
    load_subhalo=False
)

print(f"Gas loaded: {mock_data_partial.gas is not None}")
print(f"Stellar loaded: {mock_data_partial.stellar is not None}")
print(f"Subhalo loaded: {mock_data_partial.subhalo is not None}")
```

## Exploring the Data

Let's examine the structure of the loaded data:

```python
# Check shapes and types of gas data arrays
print("Gas Data Structure:")
for key, value in mock_data.gas.items():
    if isinstance(value, np.ndarray):
        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    else:
        print(f"  {key}: type={type(value)}")
```

```python
# Check shapes and types of stellar data arrays
print("Stellar Data Structure:")
for key, value in mock_data.stellar.items():
    if isinstance(value, np.ndarray):
        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    else:
        print(f"  {key}: type={type(value)}")
```

```python
# Check shapes and types of subhalo data arrays
print("Subhalo Data Structure:")
for key, value in mock_data.subhalo.items():
    if isinstance(value, np.ndarray):
        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    else:
        print(f"  {key}: type={type(value)}")
```

## Discovering Available Keys

Use the `get_available_keys()` function to see what data is available without loading everything:

```python
from kl_pipe.tng import get_available_keys

available = get_available_keys()
print("Available data keys:")
for dataset_name, keys in available.items():
    if keys is not None:
        print(f"\n{dataset_name.upper()}:")
        for key in keys:
            print(f"  - {key}")
```

## Custom Data Directory

By default, the data is loaded from `data/tng50/`. You can specify a custom location:

```python
from pathlib import Path

# Load from custom directory
custom_dir = Path("/path/to/custom/tng50/data")
# mock_data_custom = TNG50MockData(data_dir=custom_dir)

# Or with individual loaders
# gas_custom = load_gas_data(data_dir=custom_dir)
```

## Summary

The `kl_pipe.tng` module provides:

- **`TNG50MockData`**: Convenient class for loading all data
- **`load_gas_data()`**: Load gas mock data
- **`load_stellar_data()`**: Load stellar mock data  
- **`load_subhalo_data()`**: Load subhalo mock data
- **`get_available_keys()`**: Discover available data without loading

All functions support:
- Custom data directories
- Memory-efficient selective loading

## Next Steps

Now that you can load TNG50 mock data, you can:

1. Integrate it with the kinematic lensing models in `kl_pipe.model`
2. Generate synthetic observations for testing
3. Validate pipeline performance against realistic mock data
4. Develop new analysis techniques using the mock observations

See the main quickstart tutorial (`docs/tutorials/quickstart.md`) for examples of using the kinematic lensing pipeline with your data.
