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
    
    def normalization_area(self, radius_array):
        # but the first must be a cirular area  
        return 1/(np.pi * (radius_array[1:]**2 - radius_array[:-1]**2))
        

    def load_observation_data(self, exposure_time=1, radius_array = None):
        qdp_files = sorted(glob.glob(os.path.join(self.data_dir, '*.qdp')), key=self.extract_number)

        if len(qdp_files) == 0:
            raise FileNotFoundError(f"No QDP files found in {self.data_dir}")
        
        if radius_array is not None and (len(radius_array) - 1) != len(qdp_files):
            print("The length of the radius array is: " + str(len(radius_array)))
            print("The number of files are: " + str(len(qdp_files)))
            raise ValueError("The number of radius values must be equal to the number of QDP files")
        
        data = []
        # Loads the normalizations factors
        if radius_array is not None:
            norm_area = self.normalization_area(radius_array)
        else:
            norm_area = np.ones(len(qdp_files))

        for i, file in enumerate(qdp_files):
            energy, counts, error = self.read_qdp(file)
            data.append((energy, counts*exposure_time*norm_area[i], error*exposure_time*norm_area[i]))
        return data, qdp_files