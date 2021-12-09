"""
Contains classes for processing data, e.g. calculating integrated fluorescence.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
from matplotlib.pyplot import cla

import numpy as np
import pandas as pd

from .plotters import Plotter

class PreProcessor(ABC):
    """
    Abstract parent class for data processor
    """
    @abstractmethod
    def process_data(self, df: pd.DataFrame) -> None:
        """
        Processes the data
        """
        ...

@dataclass
class ProcessorPipeline(PreProcessor):
    """
    Pipeline that applies multiple data processors sequentially
    """
    processors: List[PreProcessor]
    plotters: List[Plotter] = None

    def process_data(self, df: pd.DataFrame, plot = True) -> None:
        # Process data by looping through each of the processors
        for processor in self.processors:
            processor.process_data(df)

        # Plot data if desired
        if plot and self.plotters:
            self.plot(df)

    def plot(self, df: pd.DataFrame)-> None:
        """
        Plots the data using plotters 
        """
        if not self.plotters:
            return
        
        # Loop over plotters and plot
        for plotter in self.plotters:
            plotter.plot(df)

@dataclass
class NormalizedAbsorption(PreProcessor):
    """
    Calculates normalized absorption. Fluctuations in laser power removed by using a normalizing
    photodiode. Signal without molecules present normalized to 1 by dividing by tail of signal.
    """
    background_slice: np.s_ = np.s_[-3000:] # Slice used for determining absorption signal 
                                            # background
    
    def process_data(self, df: pd.DataFrame) -> None:
        # Apply correction for intensity
        self.apply_intensity_correction(df)

        # Normalize background to 1
        self.apply_normalization(df)

    def apply_intensity_correction(self, df: pd.DataFrame) -> None:
        """
        Corrects for intensity fluctuation of laser by dividing absorption signal by 
        the signal from a normalizing photodiode. 
        """
        df["NormalizedAbsorption"] = df.AbsPD/df.AbsNormPD

    def apply_normalization(self, df: pd.DataFrame) -> None:
        """
        Normalizes the absorption signal so that when there are no molecules the signal is one 
        """
        df.NormalizedAbsorption = (df.NormalizedAbsorption
                                    /(df.NormalizedAbsorption
                                       .apply(lambda x: x[self.background_slice].mean())))

@dataclass
class IntegratedAbsorption(PreProcessor):
    """
    Calculates integrated absorption signal based on the normalized absortion signal
    """
    integration_slice: np.s_ = np.s_[10:2000] # Slice for calculating integral of absorption
    background_slice: np.s_ = np.s_[-3000:] # Slice used for determining absorption signal 
                                            # background

    def process_data(self, df: pd.DataFrame) -> None:
        df["IntegratedAbsorption"] = (df.NormalizedAbsorption
                                    .apply(self.calculate_integrated_absorption))

    def calculate_integrated_absorption(self, trace: np.ndarray) -> float:
        """
        Calculates integrated absorption for the given absorption trace
        """
        return -np.trapz(trace[self.integration_slice] - np.mean(trace[self.background_slice]))

@dataclass
class AbsorptionBigEnough(PreProcessor):
    """
    Checks if the integrated absorption signal is large enough to inculed the shot in further data
    analysis
    """
    absorption_cutoff: float = 5. # Integrated absorption needs to be bigger than this to enter
    
    def process_data(self, df: pd.DataFrame) -> None:
        df["AbsBigEnough"] = df.IntegratedAbsorption > self.absorption_cutoff

        # Store absorption cutoff in attributes of dataframe
        df.attrs["absorption_cutoff"] = self.absorption_cutoff    

@dataclass
class YAGFired(PreProcessor):
    """
    Checks if the YAG fired using data from a photodiode
    """
    YAG_cutoff: int = 250 # PD signal limit to consider YAG to have fired
    def process_data(self, df: pd.DataFrame) -> None:
        df["YAGFired"] = df.YAGPD.apply(np.max) > self.YAG_cutoff

        # Store YAG cutoff in attributes
        df.attrs["YAG_cutoff"] = self.YAG_cutoff

@dataclass
class AbsorptionON(PreProcessor):
    """
    Checks if the absorption photodiode was on by using data for the normalization photodiode
    """
    abs_norm_cutoff: int = 1000 # If the minimum signal is below this, consider absorption off
    def process_data(self, df: pd.DataFrame) -> None:
        df["AbsorptionON"] = df.AbsNormPD.apply(np.min) > self.abs_norm_cutoff

        # Store cutoff in attributes of dataframe
        df.attrs["absorption_on_cutoff"] = self.abs_norm_cutoff

@dataclass
class RotCoolON(PreProcessor):
    """
    Checks if rotational cooling laser was on by using data from photodiode
    """
    RC_pd_cutoff: int = 500 # If minimum PD signal is below this, consider RC laser to be off

    def process_data(self, df: pd.DataFrame) -> None:
        df["RCON"] = df.RCPD.apply(np.min) > self.RC_pd_cutoff

        # Store cutoff in attributes
        df.attrs["RC_on_cutoff"] = self.RC_pd_cutoff

@dataclass
class RCShutterOpen(PreProcessor):
    """
    Checks if rotational cooling shutter is supposed to be open
    """
    shutter_open_cutoff: int = 10000

    def process_data(self, df: pd.DataFrame) -> None:
        df["RCShutterOpen"] = df.RCShutter.apply(np.max) > self.shutter_open_cutoff

@dataclass
class CamDAQTimeDiff(PreProcessor):
    """
    Calculates the time difference between the time the camera and DAQ data were retrieved
    (they sometimes get out of sync)
    """
    threshold: float = 0.1 # Time differences bigger than this will get cut

    def process_data(self, df: pd.DataFrame) -> None:
        df["CamDAQTimeDiff"] = df.CameraTime - df.DAQTime
        df["TimeDiffSmallEnough"] = np.abs(df.CamDAQTimeDiff) < self.threshold

@dataclass
class IntegratedFluorescenceCam(PreProcessor):
    pass
        

