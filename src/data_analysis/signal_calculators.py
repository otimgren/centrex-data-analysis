"""
Objects for calculating signal sizes based on e.g. fluorescence images or PMT traces
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import lmfit
import numpy as np
import pandas as pd

@dataclass
class SignalResult:
    """
    Parent class for signal calculation results
    """
    signal_size: float

    @abstractmethod
    def to_df(self) -> pd.DataFrame:
        """
        Method for converting the result to a dataframe
        """
        ...

@dataclass    
class GaussianResult(SignalResult):
    """
    Class for storing results for signal sizes calculated using gaussian fitting
    """
    params: lmfit.Parameters

    def to_df(self) -> pd.DataFrame:
        """
        Converts result to a pandas dataframe
        """
        data_dict = {
            "GaussianFitFluorescenceSignal": [self.signal_size],
            "GaussianFitAmplitude": [self.params['A'].value],
            "GaussianFitCenterX": [self.params['center_x'].value],
            "GaussianFitCenterY": [self.params['center_y'].value],
            "GaussianFitSigmaX": [self.params['sigma_x'].value],
            "GaussianFitSigmaY": [self.params['sigma_y'].value],
        }
        df = pd.DataFrame(data = data_dict)
        return df

class SignalCalculator(ABC):
    """
    Abstract parent class for signal size calculators
    """
    @abstractmethod
    def calculate_signal_size(self, data: np.ndarray) -> SignalResult:
        """
        Calculates the signal size using the provided data and returns it
        """
        ...

class SignalFromGaussianFit(SignalCalculator):
    """
    Signal size calculator that fits a 2D Gaussian to a fluorescence image and returns
    the area under the gaussian
    """

    def calculate_signal_size(self, image: np.ndarray, params: lmfit.Parameters = None) -> float:
        # Fit 2D Gaussian and get result
        if not params:
            result = self.fit_2D_gaussian(image)
            params = result.params

        # Calculate the integral of the Gaussian fit
        A = params['A'].value
        sigma_x = params['sigma_x'].value
        sigma_y = params['sigma_y'].value
        integrated_gaussian = A*np.pi*sigma_x*sigma_y

        return GaussianResult(integrated_gaussian, params)


    def fit_2D_gaussian(self, image: np.ndarray, params: lmfit.Parameters = None):
        """
        Fits a 2D gaussian to data using lmfit and returns the fit result
        """
        # Get the data for the fit in the correct format
        data, x, y = self.reshape_data(image)

        # Guess parameters if not provided
        if not params:
            params = self.guess_params(data, x, y)

        # Define model
        model = self.define_model()

        # Fit the model
        result = model.fit(data, x = x, y = y, params = params, method = 'least_squares',
                           max_nfev=1000, nan_policy = 'omit')

        return result

    def guess_params(self, data: np.ndarray, x: np.array, y: np.array) -> lmfit.Parameters:
        """
        Guesses parameters for 2D gaussian fit
        """
        # Guess the parameters using a 2D gaussian without an offset and no rotation
        guessed_params = lmfit.models.Gaussian2dModel().guess(data, x = x, y = y)

        # Translate the guessed parameters into the laguage of the gaussian2D function
        params = lmfit.Parameters()
        params.add(name = 'A', value = guessed_params['height'].value, min = 0)
        params.add(name = 'center_x', value = guessed_params['centerx'], min = 0, max = 512)
        params.add(name = 'center_y', value = guessed_params['centery'], min = 0, max = 512)
        params.add(name = 'sigma_x', value = guessed_params['sigmax'], min = 10, max = 100)
        params.add(name = 'sigma_y', value = guessed_params['sigmay'], min = 10, max = 100)
        params.add(name = 'phi', value=0, min = 0, max=np.pi/4)
        params.add(name = 'C', value = 0)

        return params

    def define_model(self) -> lmfit.Model:
        """
        Defines a model to be fit using lmfit
        """
        model = lmfit.Model(self.gaussian2D, independent_vars=['x','y'])
        return model

    def reshape_data(self, image: np.ndarray) -> Tuple[np.ndarray]:
        """
        Reshapes the data to a shape that is accepted by lmfit
        """
        # Find the ranges of the x and y axes
        x_range = np.arange(image.shape[0])
        y_range = np.arange(image.shape[1])

        # Make meshgrid out of the axes
        X, Y = np.meshgrid(x_range, y_range)

        # Flatten the mehgrid arrays to get x and y coordinates for flattened image
        x_fit, y_fit = X.flatten(), Y.flatten()

        # Flatten te image
        data_fit = image.flatten()

        # Return fit cordinates and data
        return data_fit, x_fit, y_fit        

    def gaussian2D(self, x, y, A, center_x, center_y, sigma_x, sigma_y, C, phi):
        """
        Returns a Gaussian with center at (x0, y0), standard deviation σx/y, amplitude A, 
        constant offset C, and rotated by angle ϕ
        """
        xp = (x - center_x)*np.cos(phi) - (y - center_y)*np.sin(phi)
        yp = (x - center_x)*np.sin(phi) + (y - center_y)*np.cos(phi)
        R = (xp/sigma_x)**2 + (yp/sigma_y)**2
        
        return A * np.exp(-R/2) + C

