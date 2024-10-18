# Created by: Francisco Rodríguez Candón
# Last update: 06/10/2024
# mail: francandon@unizar.es

import os
import logging
import numpy as np
from astropy.io import fits

class ResponseLoader:
    def __init__(self, response_data_path, angular_bins=11):
        self.response_data_path = response_data_path
        self.angular_bins = angular_bins

    def __str__(self):
        return f"ResponseLoader({self.response_data_path})"

    @property
    def response_data_path(self):
        return self._response_data_path
    
    @response_data_path.setter
    def response_data_path(self, response_data_path):
        if not os.path.exists(response_data_path):
            raise FileNotFoundError(f"Directory not found: {response_data_path}")
        self._response_data_path = response_data_path
    
    @property
    def angular_bins(self):
        return self._angular_bins
    
    @angular_bins.setter
    def angular_bins(self, angular_bins):
        if not isinstance(angular_bins, int):
            raise TypeError("angular_bins must be an integer.")
        if angular_bins < 1:
            raise ValueError("angular_bins must be greater than zero.")
        if angular_bins != 11:
            logging.warning("The number of angular bins is expected to be 11.")
        self._angular_bins = angular_bins
    
    def load_RMF_data(self):
        """
        Load RMF (Response Matrix File) data from the specified directory.

        Parameters:
            visualize (bool): If True, additional data useful for visualization is returned.

        Returns:
            If visualize is True:
                tuple: (channel, emin, emax, energies_low, energies_high, response_data)
                    - channel (list of np.ndarray): Channels from each RMF file.
                    - emin (list of np.ndarray): Minimum energies from each RMF file.
                    - emax (list of np.ndarray): Maximum energies from each RMF file.
                    - energies_low (list of np.ndarray): Lower energy bounds from each RMF file.
                    - energies_high (list of np.ndarray): Higher energy bounds from each RMF file.
                    - response_data (np.ndarray): 3D array of response data.

            If visualize is False:
                tuple: (energies_low, energies_high, response_data)
                    - energies_low (list of np.ndarray): Lower energy bounds from each RMF file.
                    - energies_high (list of np.ndarray): Higher energy bounds from each RMF file.
                    - response_data (np.ndarray): 3D array of response data.
        """
        # Create a list of the RMF files for the annulus regions
        rmf_files = [os.path.join(self.response_data_path, f'annulus{i}.rmf') for i in range(self.angular_bins)]

        # Initialize lists to store RMF data
        energies_low = []
        energies_high = []
        responses = []

        # Load the RMF data
        for rmf_file in rmf_files:
            try:
                with fits.open(rmf_file) as rmf:
                    rmf_matrix = rmf['MATRIX'].data
                    energies_low.append(rmf_matrix['ENERG_LO'])
                    energies_high.append(rmf_matrix['ENERG_HI'])
                    responses.append(rmf_matrix['MATRIX'])
            except FileNotFoundError as e:
                logging.error(f"File not found: {rmf_file}")
                raise
            except Exception as e:
                logging.error(f"Error loading {rmf_file}: {e}")
                raise
            
    # Convert to a consistent 2D array by padding each row to the max length
        max_length = max(len(row) for response in responses for row in response)

        # Initialize a 3D array with zeros to store the response data
        response_data = np.zeros((len(responses), len(responses[0]), max_length))

        # Fill the response_data array with values
        for i, response in enumerate(responses):
            for j, row in enumerate(response):
                response_data[i, j, :len(row)] = row  # Pad with zeros if row is shorter

        return energies_low, energies_high, response_data

    def load_RMF_visualization_data(self):
        """
        Load RMF data for visualization purposes.

        Returns:
        tuple: A tuple containing the following elements:
            - Channel (list): List of channels.
            - E_min (list): List of minimum energies.
            - E_max (list): List of maximum energies.
            - Energies_low (list): List of lower energy bounds.
            - Energies_high (list): List of higher energy bounds.
            - Response_data (np.ndarray): 3D array of response data.
        """
        rmf_files = [os.path.join(self.response_data_path, f'annulus{i}.rmf') for i in range(self.angular_bins)]

        # Initialize lists to store RMF data
        channel = []
        emin = []
        emax = []
        energies_low = []
        energies_high = []
        responses = []

        # Load the RMF data
        for rmf_file in rmf_files:
            try:
                with fits.open(rmf_file) as rmf:
                    Rmf_ebounds = rmf["EBOUNDS"].data
                    channel.append(Rmf_ebounds['CHANNEL'])
                    emin.append(Rmf_ebounds['E_MIN'])
                    emax.append(Rmf_ebounds['E_MAX'])

                    Rmf_matrix = rmf['MATRIX'].data
                    energies_low.append(Rmf_matrix['ENERG_LO'])
                    energies_high.append(Rmf_matrix['ENERG_HI'])
                    responses.append(Rmf_matrix['MATRIX'])
            except FileNotFoundError as e:
                logging.error(f"File not found: {rmf_file}")
                raise
            except Exception as e:
                logging.error(f"Error loading {rmf_file}: {e}")
                raise
        
        # Convert to a consistent 2D array by padding each row to the max length
        response_data = []
        for response in responses:
            response_data.append(np.array(response))

        return channel, emin, emax, energies_low, energies_high, response_data

    def load_ARF_data(self):
        """
        Load effective area data (ARF files) from the specified directory.

        Returns:
            tuple:
                - a_eff (list of np.ndarray): List containing effective areas for each angular bin.
                - widths (list of np.ndarray): List containing bin widths for each angular bin.
                - bin_centers (list of np.ndarray): List containing bin centers for each angular bin.
        """
        # Create a list of the effective area files for the annulus regions
        eff_area_files = [
            os.path.join(self.response_data_path, f'annulus{i}.arf') 
            for i in range(self.angular_bins)
        ]

        # Load the effective area data
        ea_data = []
        for arf_file in eff_area_files:
            try:
                with fits.open(arf_file) as hdul:
                    # Assuming the data is in the second HDU (index 1)
                    data = hdul[1].data
                    ea_data.append(data)
                    logging.info(f"Successfully loaded {arf_file}.")
            except Exception as e:
                logging.error(f"Error loading {arf_file}: {e}")
                raise

        # Extract the effective area, bin widths, and bin centers
        ea_response_matrix = []
        widths = []
        bin_centers = []

        for idx, ea in enumerate(ea_data):
            available_keys = [key.upper().strip() for key in ea.columns.names]
            target_key = 'SPECRESP'

            # Check if 'SPECRESP' exists (case-insensitive)
            if target_key in available_keys:
                # Retrieve the actual key name to preserve case
                actual_key = ea.columns.names[available_keys.index(target_key)]
                ea_response = ea[actual_key]
                ea_response_matrix.append(ea_response)
                logging.debug(f"'SPECRESP' found as '{actual_key}' in annulus{idx}.rmf.")
            else:
                # If 'SPECRESP' is not found, log a warning and skip this entry
                logging.warning(
                    f"'SPECRESP' key not found in annulus{idx}.rmf. Available keys: {ea.columns.names}"
                )
                continue  # Skip to the next file

            # Calculate the bin widths
            try:
                energy_lo = ea['ENERG_LO']
                energy_hi = ea['ENERG_HI']
                width = energy_hi - energy_lo
                widths.append(width)

                # Calculate bin centers
                bin_center = energy_lo + 0.5 * width
                bin_centers.append(bin_center)
            except KeyError as ke:
                logging.error(
                    f"Missing expected energy columns in annulus{idx}.rmf: {ke}"
                )
                raise

        if not ea_response_matrix:
            logging.error("No 'SPECRESP' data found in any ARF files.")
            raise ValueError("No 'SPECRESP' data available.")

        return ea_response_matrix, widths, bin_centers