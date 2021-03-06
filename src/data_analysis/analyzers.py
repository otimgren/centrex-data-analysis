"""
Classes for further analyzing data after preprocessing
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from locale import normalize
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from .background_subtractors import BackgroundSubtractor
from .plotters import Image, Plotter, ScanParam, Switch
from .signal_calculators import SignalCalculator


class Analyzer(ABC):
    """
    Parent class for analyzers which take a dataframe with multiple rows, analyze the data
    and return a dataframe with a single row
    """

    @abstractmethod
    def analyze_data(
        self, df: pd.DataFrame, scan_param: ScanParam = None
    ) -> pd.DataFrame:
        """
        Analyzes data and returns the result of the analysis as a dataframe
        """
        ...


@dataclass
class FluorescenceImageAnalyzer(Analyzer):
    """
    Processes fluorescence images taken using a camera. Also uses information about
    absorption obtained using photodiodes and a DAQ to normalize the fluorescence.
    """

    background_subtractor: BackgroundSubtractor
    signal_calculator: SignalCalculator
    normalize: bool = True

    def analyze_data(
        self, df: pd.DataFrame, scan_param: ScanParam = None, switch: Switch = None
    ) -> pd.DataFrame:
        """
        Processes the fluorescence images in self.df and returns results as a DataFrame
        """
        # Subtract background from each image
        self.subtract_background(df)

        # Normalize each image by integrated absorption
        # self.normalize_images(df)

        # Calculate the mean image
        self.calculate_mean_image(df, scan_param, switch)

        # Normalize mean image by average integrated absorption
        if self.normalize:
            self.normalize_mean_image(df)

        # Calculate signal size
        signal_result = self.signal_calculator.calculate_signal_size(self.mean_image)

        # Convert signal result to dataframe and return it
        return signal_result.to_df()

    def subtract_background(self, df: pd.DataFrame) -> None:
        """
        Subtracts the background from the images using a BackgroundSubtractor
        """
        func = self.background_subtractor.subtract_background
        df.loc[:, "CameraData"] = df.loc[:, "CameraData"].apply(func)
        # print(df.CameraData.apply(func))

    def normalize_images(self, df: pd.DataFrame) -> None:
        """
        Normalizes fluorescence images based on integrated absorption signal by dividing each
        image by the value of the integrated absorption signal corresponding to the same
        molecule pulse as the image.
        """
        df.loc[:, "CameraData"] = (
            df.loc[:, "CameraData"].copy() / df.loc[:, "IntegratedAbsorption"].copy()
        )

    def normalize_mean_image(self, df: pd.DataFrame) -> None:
        """
        Normalizes the unnormalized mean image by dividing it by the average integrated absorption
        """
        self.mean_image.values = self.mean_image.values / df.IntegratedAbsorption.mean()

    def calculate_mean_image(
        self, df: pd.DataFrame, scan_param: ScanParam = None, switch: Switch = None
    ) -> None:
        """
        Calculates the mean of all images stored in the dataframe
        """
        mean_image = np.nanmean(np.array(list(df.loc[:, "CameraData"])), axis=0)
        self.mean_image = Image(mean_image, scan_param, switch)


@dataclass
class ParamScanAnalyzer(Analyzer):
    """
    Groups data by some scanned parameter and repeats the same analysis at each value
    of the scan parameter
    """

    scan_param: str  # Name of the scan parameter
    analyzers: List[Analyzer]  # List of analyzers that are
    plotter: Plotter = None

    def analyze_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Loop over all values of the scan parameter and analyze the data at each value
        """
        # Find all the values that the scan parameter takes
        scan_param_values = np.sort(np.unique(df[self.scan_param]))

        # Loop over scan parameter values
        df_result = pd.DataFrame()
        # print(f"Analyzing parameter scan for parameter = '{self.scan_param}'...")
        for i, value in enumerate(scan_param_values[0:None]):
            # Pick the data that corresponds to current parameter values
            data = df[df[self.scan_param] == value].copy()

            # Store the parameter value
            scan_param = ScanParam(self.scan_param, value)

            # Run all the analyzers and append to dataframe
            df_result = pd.concat(
                [df_result, self.run_analyzers(data, scan_param)], ignore_index=True
            )
            df_result.loc[i, self.scan_param] = value

        if self.plotter:
            for analyzer in self.analyzers:
                self.plotter.plot(
                    df_result, self.scan_param, analyzer.signal_calculator.signal_name
                )

        return df_result

    def run_analyzers(
        self, df: pd.DataFrame, scan_param: ScanParam = None
    ) -> pd.DataFrame:
        """
        Loop over all the analyzers in the list and merge results into a single dataframe
        """
        df_result = pd.DataFrame()
        for analyzer in self.analyzers:
            df_result = pd.concat([df_result, analyzer.analyze_data(df, scan_param)])

        return df_result


@dataclass
class SwitchingParamScanAnalyzer(Analyzer):
    """
    Groups data by some scanned parameter (e.g. frequency) and a switch (e.g. microwaves ON/OFF)
    and repeats the same analysis at each value of the scan parameter and switch
    """

    scan_param: str  # Name of the scan parameter
    switch_name: str  # Name of the dataframe column that has the switch info
    analyzers: List[Analyzer]  # List of analyzers that are
    plotter: Plotter = None

    def analyze_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Loop over all values of the scan parameter and analyze the data at each value
        """
        # Find all the values that the scan parameter takes
        scan_param_values = np.sort(np.unique(df[self.scan_param]))

        # Loop over scan parameter values
        df_result = pd.DataFrame()
        # print(f"Analyzing parameter scan for parameter = '{self.scan_param}'...")
        for i, value in enumerate(tqdm(scan_param_values[0:None], disable=True)):

            # Pick the data that corresponds to current parameter values
            data = df[df[self.scan_param] == value].copy()

            # Store the parameter value
            scan_param = ScanParam(self.scan_param, value)

            # Run all the analyzers with switch on and off and append to dataframe
            df_ON = self.run_analyzers(
                data[data[self.switch_name] == True].copy(),
                scan_param,
                Switch(self.switch_name, True),
            )
            df_OFF = self.run_analyzers(
                data[data[self.switch_name] == False].copy(),
                scan_param,
                Switch(self.switch_name, False),
            )
            df_result = pd.concat(
                [
                    df_result,
                    df_ON.merge(
                        df_OFF,
                        left_index=True,
                        right_index=True,
                        suffixes=("_ON", "_OFF"),
                    ),
                ],
                ignore_index=True,
            )

            df_result.loc[i, self.scan_param] = value

        if self.plotter:
            for analyzer in self.analyzers:
                self.plotter.plot(
                    df_result,
                    self.scan_param,
                    analyzer.signal_calculator.signal_name,
                    self.switch_name,
                )

        return df_result

    def run_analyzers(
        self, df: pd.DataFrame, scan_param: ScanParam = None, switch: Switch = None
    ) -> pd.DataFrame:
        """
        Loop over all the analyzers in the list and merge results into a single dataframe
        """
        df_result = pd.DataFrame()
        for analyzer in self.analyzers:
            df_result = pd.concat(
                [df_result, analyzer.analyze_data(df, scan_param, switch)]
            )

        return df_result

