"""
Objects used for subtracting background from camera images. Background subtractors also remove
data outside the region of interest
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

class BackgroundSubtractor:
    """
    Subtracts background from an image
    """
    @abstractmethod
    def subtract_background(self, image: np.ndarray) -> np.ndarray:
        """
        Subtracts background from an image
        """
        ...

class AcquiredBackgroundSubtractor(BackgroundSubtractor):
    """
    Subtracts a background that is based on multiple images stored in an hdf dataset
    """
    def __init__(self, background_dset, ROI: np.s_ = np.s_[0:None, 0:None]) -> None:
        super().__init__()
        self.background_dset = background_dset
        
        # Calculate background image
        self.calculate_mean_background(background_dset)
        
        # Set the region of interest
        self.ROI = ROI

    def subtract_background(self, image: np.ndarray) -> np.ndarray:
        # Subtract background 
        image_bs = image - self.mean_background

        # Set data outside the region of interest to nans
        image_roi = np.empty(image_bs.shape)
        image_roi[:] = np.nan
        image_roi[self.ROI] = image_bs[self.ROI]

        return image_roi

    def calculate_mean_background(self, df: pd.DataFrame) -> None:
        """
        Calculates the mean of the background images
        """
        data = self.background_dset['CameraData']
        self.mean_background = np.nanmean(np.array(list(data)), axis = 0).T






    

