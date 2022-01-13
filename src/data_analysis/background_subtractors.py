"""
Objects used for subtracting background from camera images
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
    Subtracts a background that is based on multiple images stored in an hdf dataset.
    """

    def __init__(self, background_dset: pd.DataFrame) -> None:
        super().__init__()
        self.background_dset = background_dset

        # Calculate background image
        self.calculate_mean_background(background_dset)

    def subtract_background(self, image: np.ndarray) -> np.ndarray:
        # Subtract background
        image_bs = image - self.mean_background

        # Return background subtracted image
        return image_bs

    def calculate_mean_background(self, df: pd.DataFrame) -> None:
        """
        Calculates the mean of the background images
        """
        data = self.background_dset["CameraData"]
        self.mean_background = np.nanmean(np.array(list(data)), axis=0)


class FittedBackgroundSubtractor(BackgroundSubtractor):
    """
    Subtracts a background based on fit around the image.
    """

    # todo
