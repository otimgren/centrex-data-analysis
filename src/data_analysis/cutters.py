"""
Cutters are used to delete undesired data
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import pandas as pd

class Cutter(ABC):
    """
    Abstract parent class for cutters
    """
    @abstractmethod
    def apply_cuts(self, df: pd.DataFrame, print_result = False) -> float:
        """
        Applies cuts to the data and returns the fraction of data cut
        """
        ...

@dataclass
class CutterPipeline(Cutter):
    """
    Applies multiple cuts sequentially
    """
    cutters: List[Cutter]

    def apply_cuts(self, df: pd.DataFrame, print_result = False) -> float:
        """
        Loop through cutters and apply cuts to dataframe
        """
        data_used = 1
        for cutter in self.cutters:
            data_used *= 1-cutter.apply_cuts(df)

        if print_result:
            print(f"Percentage of data cut: {(1-data_used)*100:.2f} %")

        return 1-data_used

class YAGFiredCutter(Cutter):
    """
    Cuts datasets where YAG didn't fire
    """
    def apply_cuts(self, df: pd.DataFrame) -> float:
        mask = df.YAGFired

        # Calculate percentage of data cut:
        data_cut = 1-(mask.sum())/len(df)

        # Cut data from dataframe
        df = df[mask].copy()

        return data_cut

class AbsorptionONCutter(Cutter):
    """
    Cuts datasets where absorption laser wasn't on
    """

    def apply_cuts(self, df: pd.DataFrame) -> float:
        mask = df.AbsorptionON

        # Calculate percentage of data cut:
        data_cut = 1-(mask.sum())/len(df)

        # Cut data from dataframe
        df = df[mask].copy()

        return data_cut

class AbsBigEnoughCutter(Cutter):
    """
    Cuts datasets where absorption signal wasn't big enough
    """

    def apply_cuts(self, df: pd.DataFrame) -> float:
        mask = df.AbsBigEnough

        # Calculate percentage of data cut:
        data_cut = 1-(mask.sum())/len(df)

        # Cut data from dataframe
        df = df[mask].copy()

        return data_cut

class RotCoolOFFShutterOpenCutter(Cutter):
    """
    Cuts data where the RC laser is off despite the RC shutter being open
    """
    def apply_cuts(self, df: pd.DataFrame, print_result=False) -> float:
        mask = ~(~df.RCON & df.RCShutterOpen)

        # Calculate percentage of data cut:
        data_cut = 1-(mask.sum())/len(df)

        # Cut data from dataframe
        df = df[mask].copy()

        return data_cut

class TimingCutter(Cutter):
    """
    Cuts datasets where time difference between camera data and DAQ data is too large
    """
    def apply_cuts(self, df: pd.DataFrame) -> float:
        mask = df.TimeDiffSmallEnough

        # Calculate percentage of data cut:
        data_cut = 1-(mask.sum())/len(df)

        # Cut data from dataframe
        df = df[mask].copy()

        return data_cut


