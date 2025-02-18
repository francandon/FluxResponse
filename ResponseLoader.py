# Created by: Francisco Rodríguez Candón
# Last update: 06/10/2024
# mail: francandon@unizar.es

import os
import re
import logging
import numpy as np
from astropy.io import fits
import glob
import numpy as np

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
    def extract_number_from_dir(self, dir_path):
        # Extracts the number from the directory name for sorting
        dir_name = os.path.basename(dir_path)
        match = re.match(r'[A-Za-z]*(\d+)', dir_name)
        if match:
            number = int(match.group(1))
            return number
        else:
            # If the pattern doesn't match, sort it at the end
            return float('inf')

    """
    This methood is the new one to load RMF data based on https://github.com/KriSun95/nustarFittingExample/blob/master/nustarFittingExample/nu_spec_code.py
    """
    def load_RMF_data(self, visualize=None):
        """
        Carga datos RMF (Response Matrix File) desde el directorio especificado.

        Parámetros:
            visualize (bool): Si es True, se retorna información adicional útil para visualización.

        Retorna:
            Si visualize es True:
                tuple: (channel, emin, emax, energies_low, energies_high, response_data)
                    - channel (list of np.ndarray): Canales de cada archivo RMF.
                    - emin (list of np.ndarray): Energías mínimas de cada archivo RMF.
                    - emax (list of np.ndarray): Energías máximas de cada archivo RMF.
                    - energies_low (list of np.ndarray): Límites inferiores de energía de cada archivo RMF.
                    - energies_high (list of np.ndarray): Límites superiores de energía de cada archivo RMF.
                    - response_data (np.ndarray): Matriz 2D de datos de respuesta.
            Si visualize es False:
                tuple: (energies_low, energies_high, response_data)
                    - energies_low (list of np.ndarray): Límites inferiores de energía de cada archivo RMF.
                    - energies_high (list of np.ndarray): Límites superiores de energía de cada archivo RMF.
                    - response_data (np.ndarray): Matriz 2D de datos de respuesta.
        """
        # Create a list of RMF file paths for each angular bin
        rmf_files = []

        # List all entries in self.response_data_path
        for entry in os.listdir(self.response_data_path):
            entry_path = os.path.join(self.response_data_path, entry)
            if os.path.isdir(entry_path):
                # This is an annulus directory, e.g., 'A0', 'A1', etc.
                dir_name = os.path.basename(entry_path)
                # The RMF file is named 'nuID.rmf' inside this directory
                # Replace 'nuID' with the actual ID or use a pattern to find the RMF file
                rmf_file_pattern = os.path.join(entry_path, '*.rmf')
                rmf_files_in_dir = glob.glob(rmf_file_pattern)
                if rmf_files_in_dir:
                    # Assuming there's only one .rmf file per annulus folder
                    rmf_file_path = rmf_files_in_dir[0]
                    rmf_files.append(rmf_file_path)
                else:
                    print(f"Warning: No RMF file found in {entry_path}")
            else:
                # Skip if not a directory
                continue

        # Sort the rmf_files list based on the annulus number extracted from the directory name
        rmf_files.sort(key=lambda x: self.extract_number_from_dir(os.path.dirname(x)))

        if len(rmf_files) == 0:
            raise FileNotFoundError(f"No RMF files found in {self.response_data_path}")

        # Initialize lists to store data from RMF files
        energies_low = []
        energies_high = []
        responses = []
        n_grp = []
        f_chan = []
        n_chan = []
        channel = []
        emin = []
        emax = []

        # Load data from each RMF file
        for rmf_file in rmf_files:
            try:
                with fits.open(rmf_file) as rmf:
                    rmf_matrix = rmf['MATRIX'].data
                    energies_low.append(rmf_matrix['ENERG_LO'])
                    energies_high.append(rmf_matrix['ENERG_HI'])
                    responses.append(rmf_matrix['MATRIX'])
                    n_grp.append(rmf_matrix['N_GRP'])
                    n_chan.append(rmf_matrix['N_CHAN'])
                    f_chan.append(rmf_matrix['F_CHAN'])
                    # Assuming 'CHANNEL', 'EMIN', 'EMAX' are present; adjust if different
                    if visualize:
                        Rmf_ebounds = rmf["EBOUNDS"].data
                        channel.append(Rmf_ebounds['CHANNEL'])
                        emin.append(Rmf_ebounds['E_MIN'])
                        emax.append(Rmf_ebounds['E_MAX'])
            except FileNotFoundError as e:
                logging.error(f"File not found: {rmf_file}")
                raise
            except Exception as e:
                logging.error(f"Error loading {rmf_file}: {e}")
                raise

        actual_response = np.zeros((len(responses), len(energies_low[0][:]), len(energies_low[0][:])))

        # Iterate over the angular bins and indices
        for angular_bin in range(len(responses)):
            for index in range(len(energies_low[0][:])):
                # Calculate the start and end indices
                inits = f_chan[angular_bin][index]
                n_chan_index = n_chan[angular_bin][index]
                start = inits
                end = inits + n_chan[angular_bin][index] - 1  # CHECK THE MINUS 1 
                # Update the actual_response with the values from responses
                counter = 0
                for start_idx, end_idx, n in zip(start, end, n_chan_index):
                    actual_response[angular_bin][index][start_idx:end_idx] = responses[angular_bin][index][counter:counter + n - 1]  # CHECK THE MINUS 1
                    counter = counter + n      

        # Converts actual_response into a numpy array
        actual_response = np.array(actual_response)
        print(actual_response.shape)
        print("RMF files", rmf_files)
        if visualize:
            return channel, emin, emax, energies_low, energies_high, actual_response
        else:   
            return energies_low, energies_high, actual_response
    
    def load_ARF_data(self):
        """
        Load effective area data (ARF files) from the specified directory.

        Returns:
            tuple:
                - a_eff (list of np.ndarray): List containing effective areas for each angular bin.
                - widths (list of np.ndarray): List containing bin widths for each angular bin.
                - bin_centers (list of np.ndarray): List containing bin centers for each angular bin.
        """
        # Initialize lists to store data from ARF files
        ea_data = []
        arf_files = []

        # List all entries in self.response_data_path
        for entry in os.listdir(self.response_data_path):
            entry_path = os.path.join(self.response_data_path, entry)
            if os.path.isdir(entry_path):
                # This is an annulus directory, e.g., 'A0', 'A1', etc.
                dir_name = os.path.basename(entry_path)
                # The ARF file is named 'nuID.arf' inside this directory
                # Use a pattern to find the ARF file
                arf_file_pattern = os.path.join(entry_path, '*.arf')
                arf_files_in_dir = glob.glob(arf_file_pattern)
                if arf_files_in_dir:
                    # Assuming there's only one .arf file per annulus folder
                    arf_file_path = arf_files_in_dir[0]
                    arf_files.append(arf_file_path)
                else:
                    print(f"Warning: No ARF file found in {entry_path}")
            else:
                # Skip if not a directory
                continue

        # Sort the arf_files list based on the annulus number extracted from the directory name
        arf_files.sort(key=lambda x: self.extract_number_from_dir(os.path.dirname(x)))

        if len(arf_files) == 0:
            raise FileNotFoundError(f"No ARF files found in {self.response_data_path}")

        # Load the effective area data
        for arf_file in arf_files:
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
                logging.debug(f"'SPECRESP' found as '{actual_key}' in ARF file at index {idx}.")
            else:
                # If 'SPECRESP' is not found, log a warning and skip this entry
                logging.warning(
                    f"'SPECRESP' key not found in ARF file at index {idx}. Available keys: {ea.columns.names}"
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
                    f"Missing expected energy columns in ARF file at index {idx}: {ke}"
                )
                raise

        if not ea_response_matrix:
            logging.error("No 'SPECRESP' data found in any ARF files.")
            raise ValueError("No 'SPECRESP' data available.")
        return ea_response_matrix, widths, bin_centers
    
        """
    This method is the OLD ONE.
    
    
    def load_RMF_data(self):

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

        # Create a list of the RMF files for the annulus regions
        rmf_files = [os.path.join(self.response_data_path, f'annulus{i}.rmf') for i in range(self.angular_bins)]

        # Initialize lists to store RMF data
        energies_low = []
        energies_high = []
        responses = []
        n_grp = []
        n_chan = []

        # Load the RMF data
        for rmf_file in rmf_files:
            try:
                with fits.open(rmf_file) as rmf:
                    rmf_matrix = rmf['MATRIX'].data
                    energies_low.append(rmf_matrix['ENERG_LO'])
                    energies_high.append(rmf_matrix['ENERG_HI'])
                    responses.append(rmf_matrix['MATRIX'])
                    n_grp.append(rmf_matrix['N_GRP'])
                    n_chan.append(rmf_matrix['N_CHAN'])
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
    """