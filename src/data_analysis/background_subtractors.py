"""
Objects used for subtracting background from camera images
"""
from abc import ABC, abstractmethod

import numpy as np

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


    

