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
    
    def extract_number(self, filename):
            # Modify the regex to extract only numbers, but prioritize integers for sorting
            match = re.search(r'(\d+)', filename)
            if match:
                return int(match.group(1))  # Return the number as an integer for proper sorting
            return float('inf')  # Return a high value for filenames without numbers, to sort them last
    def load_observation_data(self, exposure_time=1):
        qdp_files = sorted(glob.glob(os.path.join(self.data_dir, '*.qdp')), key=self.extract_number)

        data = []
        for file in qdp_files:
            energy, counts, error = self.read_qdp(file)
            data.append((energy, counts*exposure_time, error*exposure_time))
        return data, qdp_files