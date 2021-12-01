"""
Contains classes for retrieving data from file (or wherever it's stored)
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Union

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
    run_name: str # Name of the run whose data is needed
    scan_param: str = None #
    NI_DAQ_path: str = 'readout'
    def retrieve_data(self) -> pd.DataFrame:
        pass

    def retrieve_camera_data(self, filepath: Union[Path, str], run_name: str,
                             camera_path: str) -> pd.DataFrame:
        """
        Loads camera data from hdf file.
        """
        # Initialize containers for camera images and their timestamps
        camera_data = []
        camera_time = []

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

    def retrieve_NI_DAQ_data(self, filepath: Union[Path, str], run_name: str, NI_DAQ_path: str,
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

        # Determine path to data within the hdf file
        data_path = f"{run_name}/{NI_DAQ_path}/PXIe-5171"

        # Open hdf file
        with h5py.File(filepath, 'r') as f:
            # Loop over camera images (1 image per molecule pulse)
            for dataset_name in f[data_path]:
                if 'events' not in dataset_name:
                    n = int(dataset_name.split('_')[-1])
                    DAQ_data.append(f[data_path][dataset_name][()])
                    DAQ_time.append(f[data_path][dataset_name].attrs[f'timestamp'])
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





