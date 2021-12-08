"""
Contains classes for retrieving data from file (or wherever it's stored)
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import h5py
import pandas as pd
from pandas.core.frame import DataFrame

@dataclass
class Retriever(ABC):
    """
    Abstract parent class for data retrievers
    """
    @abstractmethod
    def retrieve_data(self) -> pd.DataFrame:
        """
        Retrieves data from file
        """

class SPARetriever(Retriever):
    """
    Retriever used with SPA test data
    """
    def retrieve_data(self, filepath: Union[Path, str], run_name: Union[str, int],
                      camera_path: str = 'camera_test', NI_DAQ_path: str = 'readout', 
                      scan_param: str = None, muwave_shutter = True,
                      scan_param_new_name: str = None) -> pd.DataFrame:
        """
        Reterieves SPA test data from file
        """
        # Retrieve camera data
        df_CAM = self.retrieve_camera_data(filepath, run_name, camera_path)

        # Retrieve DAQ data
        df_DAQ = self.retrieve_NI_DAQ_data(filepath, run_name, NI_DAQ_path, scan_param, muwave_shutter)

        # Merge dataframes
        df = df_CAM.merge(df_DAQ, left_index=True, right_index=True)

        # If needed, give scan parameter a new name
        if scan_param_new_name:
            df.rename(mapper = {scan_param : scan_param_new_name}, inplace = True, axis = 1)

        # Return merged dataframe
        return df

    def retrieve_camera_data(self, filepath: Union[Path, str], run_name: Union[str, int],
                             camera_path: str) -> pd.DataFrame:
        """
        Loads camera data from hdf file.
        """
        # Initialize containers for camera images and their timestamps
        camera_data = []
        camera_time = []

        # If run_name given as an index, get the string version
        if type(run_name) == int:
            run_name = self.get_run_names(filepath)[run_name]

        # Determine the path to data within the hdf file
        data_path = f"{run_name}/{camera_path}/PIProEM512Excelon"

        # Open hdf file
        with h5py.File(filepath, 'r') as f:
            # Loop over camera images (1 image per molecule pulse)
            for dataset_name in f[data_path]:
                if 'events' not in dataset_name:
                    n = int(dataset_name.split('_')[-1])
                    camera_data.append(f[data_path][dataset_name][()])
                    camera_time.append(f[data_path][dataset_name].attrs[f'timestamp'])

        # Convert lists into a dataframe and return it
        dataframe = pd.DataFrame(data = {"CameraTime" :camera_time, "CameraData": camera_data})
        return dataframe

    def retrieve_NI_DAQ_data(self, filepath: Union[Path, str], run_name: Union[str, int], NI_DAQ_path: str,
                             scan_param: str = None, muwave_shutter = True) -> pd.DataFrame:
        """
        Retrieves data obtained using the NI5171 PXIe DAQ
        """
        # Define which channel on DAQ corresponds to which data
        yag_ch = 0 # Photodiode observing if YAG fired
        abs_pd_ch = 2 # Photodiode observing absorption outside cold cell
        abs_pd_norm_ch = 3 # Photodiode to normalize for laser intensity fluctuations in absorption
        rc_shutter_ch = 4 # Tells if rotational cooling laser shutter is open or closed
        rc_pd_ch = 5 # Photodiode for checking that rotaional cooling is on
        muwave_shutter_ch = 6 # Tells if SPA microwaves are on or off 
        
        # Initialize containers for data
        DAQ_data = []
        DAQ_time = []
        DAQ_attrs = []

        # If run_name given as an index, get the string version
        if type(run_name) == int:
            run_name = self.get_run_names(filepath)[run_name]
        
        # Determine path to data within the hdf file
        data_path = f"{run_name}/{NI_DAQ_path}/PXIe-5171"

        # Open hdf file
        with h5py.File(filepath, 'r') as f:
            # Loop over camera images (1 image per molecule pulse)
            for dataset_name in f[data_path]:
                if 'events' not in dataset_name:
                    n = int(dataset_name.split('_')[-1])
                    DAQ_data.append(f[data_path][dataset_name][()])
                    DAQ_time.append(f[data_path][dataset_name].attrs['ch0 : timestamp'])
                    DAQ_attrs.append({key:value for key, value 
                                        in f[data_path][dataset_name].attrs.items()})

        # Convert lists to dataframes
        data_dict = {
            "YAGPD": [dataset[:, yag_ch] for dataset in DAQ_data],
            "AbsPD": [dataset[:, abs_pd_ch] for dataset in DAQ_data],
            "AbsNormPD": [dataset[:, abs_pd_norm_ch] for dataset in DAQ_data],
            "RCShutter": [dataset[:, rc_shutter_ch] for dataset in DAQ_data],
            "RCPD": [dataset[:, rc_pd_ch] for dataset in DAQ_data],
            "DAQTime": DAQ_time
            }

        # If microwave shutter was used, need that
        if muwave_shutter:
            data_dict["MicrowaveShutter"] = [dataset[:, muwave_shutter_ch] for dataset in DAQ_data]

        # If scan parameter was specified, get data for that
        if scan_param:
            data_dict[scan_param] = [dataset[scan_param] for dataset in DAQ_attrs]

        # Convert dictionary to dataframe and return it
        dataframe = pd.DataFrame(data = data_dict)
        return dataframe

    def get_run_names(self, filepath: Union[Path, str]) -> List[str]:
        """
        Gets the names of the datasets stored in a given file
        """
        with h5py.File(filepath, 'r') as f:
            keys = list(f.keys())
        
        return keys

    def print_run_names(self, filepath: Union[Path, str]) -> None:
        """
        Prints the names of the datasets stored in the given file
        """
        # Get dataset names
        keys = self.get_run_names(filepath)

        # Print dataset names
        print("Dataset names:")
        for i, key in enumerate(keys):
            print(f"{i} -- {key}")




