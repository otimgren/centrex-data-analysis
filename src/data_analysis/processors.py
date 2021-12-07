"""
Classes for further processing data after preprocessing
"""
from dataclasses import dataclass

from .preprocessors import Processor

@dataclass
class FluorescenceImageProcessor(Processor):
    """
    Processes fluorescence images taken using a camera. Also uses information about 
    absorption obtained using photodiodes and a DAQ.
    """
    background_subtractor: 


