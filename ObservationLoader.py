import re
import os 
import glob
import pandas as pd
import numpy as np

class ObservationLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def __str__(self):
        return f"ObservationLoader({self.data_dir})"
    
    @property
    def data_dir(self):
        return self._data_dir
    
    @data_dir.setter
    def data_dir(self, data_dir):
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Directory not found: {data_dir}")
        self._data_dir = data_dir

    def read_qdp(self, file_path):
        data = pd.read_csv(file_path, comment='!', sep='\s+', header=None, skiprows=2)
        energy = data[0]
        counts = data[2]
        error = data[3]
        return energy, counts, error
    
    def extract_number_from_dir(self, filepath):
        # Extracts the letter(s) and number from the directory name for sorting
        dir_name = os.path.basename(os.path.dirname(filepath))
        match = re.match(r'([A-Z]+)(\d+)', dir_name, re.IGNORECASE)
        if match:
            letter = match.group(1).upper()
            number = int(match.group(2))
            # Return a tuple for sorting: first by letter, then by number
            return (letter, number)
        else:
            # If the pattern doesn't match, sort it at the end
            return ('', float('inf'))
        
    def normalization_area(self, radius_array):
        # but the first must be a cirular area  
        return 1/(np.pi * (radius_array[1:]**2 - radius_array[:-1]**2))
        
    def load_observation_data(self, exposure_time=1, radius_array=None):
            """
            Load observation data from .qdp files in the specified data directory.

            Args:
                exposure_time (float): The exposure time to normalize the counts.
                radius_array (np.ndarray): Array of radius values for normalization.

            Returns:
                tuple: A tuple containing the data list and the list of qdp file paths.
            """
            data_dir = self.data_dir
            qdp_files = []

            # List all entries in data_dir
            for entry in os.listdir(data_dir):
                entry_path = os.path.join(data_dir, entry)
                if os.path.isdir(entry_path):
                    dir_name = os.path.basename(entry_path)
                    # Expected file name is 'annulus' + dir_name + '.qdp'
                    qdp_file_name = f'annulus{dir_name}.qdp'
                    qdp_file_path = os.path.join(entry_path, qdp_file_name)
                    if os.path.isfile(qdp_file_path):
                        qdp_files.append(qdp_file_path)
                    else:
                        print(f"Warning: Expected QDP file {qdp_file_name} not found in {entry_path}")
                else:
                    # Skip if not a directory
                    continue

            # Sort the qdp_files list based on the annulus number extracted from the directory name
            qdp_files.sort(key=self.extract_number_from_dir)

            if len(qdp_files) == 0:
                raise FileNotFoundError(f"No QDP files found in {self.data_dir}")

            # Now proceed as before to load the data
            data = []
            # Load the normalization factors
            if radius_array is not None:
                norm_area = self.normalization_area(radius_array)
            else:
                norm_area = np.ones(len(qdp_files))

            for i, file in enumerate(qdp_files):
                energy, counts, error = self.read_qdp(file)
                data.append((energy, counts * exposure_time * norm_area[i], error * exposure_time * norm_area[i]))
            return data, qdp_files
