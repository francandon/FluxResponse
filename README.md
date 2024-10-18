# FluxResponse

This folder contains a package to compute the expected counts at the detector, by integrating the theoretical flux with the response files and then rebin the results. 

## Files

# 1. **ObservationLoader.py**:
 Contains the `ObservationLoader` class for reading `.qdp` files and loading observation data.

## Methods

### `__init__(self, data_dir)`
- Initializes the `ObservationLoader` with the specified directory. Raises `FileNotFoundError` if the directory does not exist.

### `read_qdp(self, file_path)`
- Reads a QDP file and returns energy, counts, and error values.

### `extract_number(self, filename)`
- Extracts the first number from a filename for sorting. Returns a high value if no number is found.

### `load_observation_data(self, exposure_time=1)`
- Loads all `.qdp` files, scales the counts and errors by `exposure_time`, and returns the processed data.

## Example
```python
loader = ObservationLoader('/path/to/data')
data, files = loader.load_observation_data(exposure_time=10)
```

# 2. **ResponseLoader.py**:
 Contains the `ResponseLoader` class for loading `.rmf` and `.arf` files.

## Methods

### `__init__(self, response_data_path, angular_bins=11)`
- Initializes the loader with a directory path and number of angular bins. Raises `FileNotFoundError` for invalid paths.

### `load_RMF_data(self)`
- Loads RMF data (energies and response matrices) for each angular bin. Returns energies and response data as arrays.

### `load_RMF_visualization_data(self)`
- Loads additional RMF data (channels, energies, responses) for visualization purposes.

### `load_ARF_data(self)`
- Loads ARF data, extracting effective areas, bin widths, and bin centers.

## Example
```python
loader = ResponseLoader('/path/to/data', angular_bins=11)
energies_low, energies_high, response_data = loader.load_RMF_data()
```

# 3. **TheoreticalFlux.py**:
 Contains the `TheoreticalFlux` class and Numba-optimized functions for flux rebinning and count computation.

## Methods

### `compute_expected_counts(self, exposure_time=1.0)`
- Computes expected counts in each detector bin using ARF and RMF data.

### `rebin_flux(self, arf_energy)`
- Rebins the flux array to match ARF energy bins using linear interpolation.

### `compute_trapz_weights(self, energies)`
- Computes trapezoidal integration weights for energy bins.

### `compute_expected_counts_numba(flux, arf, rmf, weights, exposure_time)`
- Numba-optimized function to compute expected counts per detector bin.

## Example
```python
flux_model = TheoreticalFlux(response_loader, '/path/to/data')
counts, energies = flux_model.compute_expected_counts(exposure_time=1000)
```