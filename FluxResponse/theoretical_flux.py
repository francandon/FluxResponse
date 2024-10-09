# Created by: Francisco Rodríguez Candón
# Last update: 06/10/2024
# Email: francandon@unizar.es

import os
import logging
import numpy as np
from numba import njit, prange

from .load_responses import ResponseLoader  # Relative import

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def channel_to_energy(channel): 
    return channel * 0.04 + 1.6  # keV

@njit(parallel=True, fastmath=True)
def compute_expected_counts_numba(flux, arf, rmf, weights, exposure_time):
    """
    Numba-optimized function to compute expected counts.
    
    Parameters
    ----------
    flux : np.ndarray
        Flux array. Shape: (N_energy_bins,)
    arf : np.ndarray
        ARF array. Shape: (N_energy_bins,)
    rmf : np.ndarray
        Response matrix for one annulus. Shape: (N_detector_bins, N_energy_bins)
    weights : np.ndarray
        Integration weights. Shape: (N_energy_bins,)
    exposure_time : float
        Exposure time in seconds.
    
    Returns
    -------
    np.ndarray
        Expected counts per detector bin. Shape: (N_detector_bins,)
    """
    effective_flux = arf * flux * weights
    num_detector_bins = rmf.shape[0]
    expected_counts = np.zeros(num_detector_bins, dtype=flux.dtype)
    
    for i in prange(num_detector_bins):
        expected_counts[i] = exposure_time * np.dot(rmf[i, :], effective_flux)
    return expected_counts

@njit(parallel=True, fastmath=True)
def rebin_flux_numba(flux, flux_energy, arf_energy, rebinned_flux):
    """
    Numba-optimized function to rebin the flux array to match ARF energy bins using linear interpolation.
    
    Parameters
    ----------
    flux : np.ndarray
        Original flux array. Shape: (Ng, Nm, N_annulus, N_flux_bins)
    flux_energy : np.ndarray
        Original energy bins. Shape: (N_flux_bins,)
    arf_energy : np.ndarray
        Target energy bins for rebinned flux. Shape: (N_new_bins,)
    rebinned_flux : np.ndarray
        Array to store the rebinned flux. Shape: (Ng, Nm, N_annulus, N_new_bins)
    """
    Ng, Nm, N_annulus, N_flux_bins = flux.shape
    N_new_bins = arf_energy.shape[0]
    
    for i in prange(Ng):
        for j in range(Nm):
            for k in range(N_annulus):
                for e in range(N_new_bins):
                    x = arf_energy[e]
                    if x < flux_energy[0] or x > flux_energy[-1]:
                        rebinned_flux[i, j, k, e] = 0.0
                    else:
                        # Find the right index using binary search
                        idx = np.searchsorted(flux_energy, x) - 1
                        if idx < 0 or idx >= N_flux_bins - 1:
                            rebinned_flux[i, j, k, e] = 0.0
                        else:
                            # Linear interpolation
                            x0 = flux_energy[idx]
                            x1 = flux_energy[idx + 1]
                            y0 = flux[i, j, k, idx]
                            y1 = flux[i, j, k, idx + 1]
                            t = (x - x0) / (x1 - x0)
                            rebinned_flux[i, j, k, e] = y0 * (1.0 - t) + y1 * t

class TheoreticalFlux:
    def __init__(self,
                 response_loader: ResponseLoader,
                 flux_data_path: str,
                 flux_filename: str = "FluxArrayRev.npy",
                 energy_filename: str = "EnGammaRev.npy",
                 ggrid_filename: str = "ggrid.npy",
                 mgrid_filename: str = "mgrid.npy"):
        self.response_loader = response_loader
        self.flux_data_path = flux_data_path
        self.flux_filename = flux_filename
        self.energy_filename = energy_filename
        self.ggrid_filename = ggrid_filename
        self.mgrid_filename = mgrid_filename

        # Load the data with efficient data types and ensure C-contiguity
        self.flux = np.ascontiguousarray(
            np.load(os.path.join(self.flux_data_path, self.flux_filename)), dtype=np.float32
        )
        self.flux_energy = np.ascontiguousarray(
            np.load(os.path.join(self.flux_data_path, self.energy_filename)), dtype=np.float32
        )
        self.ggrid = np.ascontiguousarray(
            np.load(os.path.join(self.flux_data_path, self.ggrid_filename)), dtype=np.float32
        )
        self.mgrid = np.ascontiguousarray(
            np.load(os.path.join(self.flux_data_path, self.mgrid_filename)), dtype=np.float32
        )

    def __str__(self):
        return f"TheoreticalFlux(flux_data_path='{self.flux_data_path}')"

    # Properties with getters and setters
    @property
    def response_loader(self) -> ResponseLoader:
        return self._response_loader

    @response_loader.setter
    def response_loader(self, value: ResponseLoader):
        if not isinstance(value, ResponseLoader):
            raise TypeError("response_loader must be a ResponseLoader object.")
        self._response_loader = value

    @property
    def flux_data_path(self) -> str:
        return self._flux_data_path

    @flux_data_path.setter
    def flux_data_path(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Directory not found: {path}")
        self._flux_data_path = path

    @property
    def flux(self) -> np.ndarray:
        return self._flux

    @flux.setter
    def flux(self, value: np.ndarray):
        if not isinstance(value, np.ndarray):
            raise TypeError("flux must be a NumPy array.")
        if value.ndim != 4:
            raise ValueError("flux must be a 4D array.")
        self._flux = np.ascontiguousarray(value, dtype=np.float32)

    @property
    def flux_energy(self) -> np.ndarray:
        return self._flux_energy

    @flux_energy.setter
    def flux_energy(self, value: np.ndarray):
        if not isinstance(value, np.ndarray):
            raise TypeError("flux_energy must be a NumPy array.")
        self._flux_energy = np.ascontiguousarray(value, dtype=np.float32)

    @property
    def ggrid(self) -> np.ndarray:
        return self._ggrid

    @ggrid.setter
    def ggrid(self, value: np.ndarray):
        if not isinstance(value, np.ndarray):
            raise TypeError("ggrid must be a NumPy array.")
        self._ggrid = np.ascontiguousarray(value, dtype=np.float32)

    @property
    def mgrid(self) -> np.ndarray:
        return self._mgrid

    @mgrid.setter
    def mgrid(self, value: np.ndarray):
        if not isinstance(value, np.ndarray):
            raise TypeError("mgrid must be a NumPy array.")
        self._mgrid = np.ascontiguousarray(value, dtype=np.float32)

    def rebin_flux(self, arf_energy: np.ndarray) -> np.ndarray:
        """
        Rebin the flux array to match ARF energy bins using a Numba-optimized function.
        
        Parameters
        ----------
        arf_energy : np.ndarray
            Target energy bins for rebinned flux. Shape: (N_new_bins,)
        
        Returns
        -------
        np.ndarray
            Rebinned flux array. Shape: (Ng, Nm, N_annulus, N_new_bins)
        """
        Ng, Nm, N_annulus, N_flux_bins = self.flux.shape
        N_new_bins = len(arf_energy)

        # Initialize the rebinned_flux array
        rebinned_flux = np.zeros((Ng, Nm, N_annulus, N_new_bins), dtype=np.float32)

        # Perform interpolation using the Numba-optimized function
        rebin_flux_numba(self.flux, self.flux_energy, arf_energy, rebinned_flux)

        return rebinned_flux

    def compute_trapz_weights(self, energies: np.ndarray) -> np.ndarray:
        """
        Compute the trapezoidal integration weights for the given energy bins.
        
        Parameters
        ----------
        energies : np.ndarray
            The energy bin centers or edges.
        
        Returns
        -------
        np.ndarray
            Integration weights for the trapezoidal rule.
        """
        energy_diffs = np.diff(energies)
        weights = np.zeros_like(energies, dtype=energies.dtype)
        weights[1:-1] = (energy_diffs[:-1] + energy_diffs[1:]) / 2.0
        weights[0] = energy_diffs[0] / 2.0
        weights[-1] = energy_diffs[-1] / 2.0
        return weights

    def compute_expected_counts(self, exposure_time: float = 1.0) -> np.ndarray:
        """
        Compute the expected counts in each detector energy bin using the Numba-optimized method.
        
        Parameters
        ----------
        exposure_time : float, optional
            Exposure time in seconds (default is 1.0).
        
        Returns
        -------
        np.ndarray
            Array of expected counts per detector bin.
            Shape: (Ng, Nm, N_annulus, N_detector_bins)
        """
        logger.info("Loading ARF and RMF data...")
        # Load ARF and RMF data
        arf, _, arf_energy = self.response_loader.load_ARF_data()
        _, _, rmf = self.response_loader.load_RMF_data()

        # Create the energy array
        channels = np.arange(1, 4097)
        energy_all = channel_to_energy(channels).astype(np.float32)
        # Obtain the indices for energy = 30 keV and 70 keV
        index_30 = np.argmin(np.abs(energy_all - 30))
        index_70 = np.argmin(np.abs(energy_all - 70))
        energies = energy_all[index_30:index_70]  # Energies between 30 keV and 70 keV

        # Convert to numpy arrays with efficient data types and ensure C-contiguity
        arf_energy = np.ascontiguousarray(np.array(arf_energy, dtype=np.float32))  # Shape: (N_annulus, N_arf_bins)
        arf = np.ascontiguousarray(np.array(arf, dtype=np.float32))                # Shape: (N_annulus, N_arf_bins)
        rmf = np.ascontiguousarray(np.array(rmf, dtype=np.float32))                # Shape: (N_annulus, N_detector_bins, N_arf_bins)

        # Constrain the energy range between 30 keV and 70 keV for the ARF and RMF arrays
        arf = arf[:, index_30:index_70]  # Shape: (N_annulus, N_energies)
        rmf = rmf[:, index_30:index_70, index_30:index_70]  # Shape: (N_annulus, N_detector_bins, N_energies)
   
        logger.info("Rebinning flux to match ARF energy bins...")
        flux = self.rebin_flux(energies)  # Shape: (Ng, Nm, N_annulus, N_arf_bins)

        # Determine grid sizes
        Ng = len(self.ggrid)
        Nm = len(self.mgrid)
        N_energies = len(energies)
        N_annulus = arf.shape[0] 

        logger.info("Computing trapezoidal integration weights...")
        # Precompute the trapezoidal integration weights once outside the loop
        weights = self.compute_trapz_weights(energies)  # Shape: (N_energies,)

        # Initialize expected counts array
        expected_counts = np.zeros((Ng, Nm, N_annulus, rmf.shape[1]), dtype=np.float32)

        logger.info("Starting computation of expected counts...")
        for idx_g in prange(Ng):
            coupling = self.ggrid[idx_g]
            for idx_m in range(Nm):
                mass = self.mgrid[idx_m]
                logger.debug(f'Processing: Coupling={coupling}, Mass={mass}')
                for i in range(N_annulus):
                    counts = compute_expected_counts_numba(
                        flux[idx_g, idx_m, i, :],
                        arf[i, :],
                        rmf[i, :, :],
                        weights,
                        exposure_time=exposure_time
                    )
                    expected_counts[idx_g, idx_m, i, :] = counts

        logger.info("Completed computation of expected counts.")
        # Changes the self.flux to the nuew flux
        self.flux = expected_counts
        self.flux_energy = energies
        return expected_counts, energies
    def resizing_energy(self, energy_min : float = 30, energy_max : float = 70) -> np.ndarray:

        # Create the energy array
        channels = np.arange(1, 4097)
        energy_all = channel_to_energy(channels).astype(np.float32)

        # Obtain the indices for energy = 30 keV and 70 keV
        index_30 = np.argmin(np.abs(energy_all - energy_min))
        index_70 = np.argmin(np.abs(energy_all - energy_max))
        energies = energy_all[index_30:index_70]  # Energies between 30 keV and 70 keV
        return energies, index_30, index_70

        

    def compute_expected_counts_numpy(self, exposure_time: float = 1.0) -> np.ndarray:
        """
        Compute the expected counts in each detector energy bin using NumPy's trapezoidal integration.
        This method is not optimized for speed and is intended for validation purposes.
        
        Parameters
        ----------
        exposure_time : float, optional
            Exposure time in seconds (default is 1.0).
        
        Returns
        -------
        np.ndarray
            Array of expected counts per detector bin.
            Shape: (Ng, Nm, N_annulus, N_detector_bins)
        """
        logger.info("Starting NumPy-based computation of expected counts...")
        
        # Load ARF and RMF data
        arf, _, arf_energy = self.response_loader.load_ARF_data()
        _, _, rmf = self.response_loader.load_RMF_data()

        # Create the energy array
        energies, index_min, index_max = self.resizing_energy()  # Energies between 30 keV and 70 keV
        self.flux_energy = energies

        # Convert to numpy arrays with efficient data types
        arf_energy = np.array(arf_energy, dtype=np.float32)  # Shape: (N_annulus, Energy_bins)
        arf = np.array(arf, dtype=np.float32)                # Shape: (N_annulus, ARF_values)
        rmf = np.array(rmf, dtype=np.float32)                # Shape: (N_annulus, N_energy_index, N_energy_bins)

        # Constrain the energy range between 30 keV and 70 keV for the ARF and RMF arrays
        arf = arf[:, index_min:index_max]  # Shape: (N_annulus, N_energies)
        rmf = rmf[:, index_min:index_max, index_min:index_max]  # Shape: (N_annulus, N_detector_bins, N_energies)
   
        logger.info("Rebinning flux to match ARF energy bins...")
        flux = self.rebin_flux(energies)  # Shape: (Ng, Nm, N_annulus, N_arf_bins)

        # Determine grid sizes
        Ng = len(self.ggrid)
        Nm = len(self.mgrid)
        N_annulus = arf.shape[0] 

        # Initialize expected counts array
        expected_counts_numpy = np.zeros((Ng, Nm, N_annulus, arf.shape[1]), dtype=np.float32)

        logger.info("Starting computation of expected counts using NumPy...")
        for idx_g in range(Ng):
            coupling = self.ggrid[idx_g]
            for idx_m in range(Nm):
                mass = self.mgrid[idx_m]
                logger.debug(f'Processing: Coupling={coupling}, Mass={mass}')
                for i in range(N_annulus):
                    effective_flux = arf[i, :] * flux[idx_g, idx_m, i, :]
                    counts = exposure_time * np.trapz(rmf[i, :, :] * effective_flux, energies, axis=1)  # Shape: (N_detector_bins,)

                    expected_counts_numpy[idx_g, idx_m, i, :] = counts

        logger.info("Completed NumPy-based computation of expected counts.")
        return expected_counts_numpy, energies

    def rebin_expected_counts_custom(self, expected_counts: np.ndarray, new_bounds: np.ndarray, energies: np.ndarray) -> np.ndarray:
        """
        Rebin the expected counts array by resizing the energy dimension into smaller predefined energy bins
        using specified new_bounds.
        
        Parameters
        ----------
        expected_counts : np.ndarray
            The original expected counts array. Shape: (Ng, Nm, N_annulus, N_energy_bins)
        new_bounds : np.ndarray
            The new energy bin boundaries. Shape: (new_energy_bins + 1,)
        
        Returns
        -------
        np.ndarray
            The rebinned expected counts array. Shape: (Ng, Nm, N_annulus, new_energy_bins)
        
        Raises
        ------
        ValueError
            If new_bounds is not a 1D array, not sorted, or if new_bounds extend beyond original energy bins.
        """
        if new_bounds.ndim != 1:
            raise ValueError("new_bounds must be a 1D array of energy bin boundaries.")
        
        if not np.all(new_bounds[:-1] < new_bounds[1:]):
            raise ValueError("new_bounds must be sorted in ascending order.")
        
        min_new = new_bounds[0]
        max_new = new_bounds[-1]
        
        if min_new < energies[0] or max_new > energies[-1]:
            print("The in_nes and the energie[0]: ", min_new, energies[0])
            # raise ValueError("new_bounds must be within the range of flux_energy.")
        
        Ng, Nm, N_annulus, N_energy_bins = expected_counts.shape
        new_energy_bins = len(new_bounds) - 1
        new_expected_counts = np.zeros((Ng, Nm, N_annulus, new_energy_bins), dtype=expected_counts.dtype)
        
        logger.info("Starting custom rebinning of expected counts...")
        for j in range(new_energy_bins):
            lower = new_bounds[j]
            upper = new_bounds[j + 1]
            # Create a mask for the original energy bins within the new bin boundaries
            mask = (energies >= lower) & (energies < upper)
            # Sum the counts over the selected energy bins
            new_expected_counts[:,:,:,j] = expected_counts[:,:,:,mask].sum(axis=3)
        
        logger.info("Completed custom rebinning of expected counts.")
        return new_expected_counts
