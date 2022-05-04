"""
Classes used for visualizing data
"""
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import lmfit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

PLOT_DIR = Path("./saved_plots")


@dataclass
class ScanParam:
    name: str
    value: float


@dataclass
class Switch:
    name: str
    value: float


@dataclass
class Image:
    values: np.ndarray  # Array with values for image brightness
    scan_param: ScanParam = None  # Info about scan parameter if needed
    switch: Switch = None


@dataclass
class Plotter(ABC):
    """
    Abstract parent class for plotters
    """

    @abstractmethod
    def plot(self, df: pd.DataFrame) -> None:
        """
        Plots the needed plots
        """
        ...


@dataclass
class PlotterDefault(Plotter):
    figsize: Tuple[int] = (16, 9)
    label_fontsize: int = 16

    def __post_init__(self):
        self.fig = None
        self.ax = None


@dataclass
class PreProcessorPlotter(PlotterDefault):
    """
    Used for making a plot after data has been preprocessed
    """

    ncol: int = 2
    nrow: int = 2

    def plot(self, df: pd.DataFrame) -> None:
        self.setup_plots()
        self.plot_absorption(df)
        self.plot_YAG_firing(df)
        self.plot_absorption_ON(df)
        self.plot_RC_ON(df)
        plt.show()

    def setup_plots(self):
        """
        Sets up the figure and axes
        """
        self.fig, self.ax = plt.subplots(self.ncol, self.nrow, figsize=self.figsize)

    def plot_absorption(self, df: pd.DataFrame) -> None:
        """
        Plots the sizes of integrated absorption signals on a histogram
        """
        # Plot for absorption cutoff
        df.IntegratedAbsorption[df.YAGFired & df.AbsorptionON].hist(
            bins=20, ax=self.ax[0, 0]
        )
        df.IntegratedAbsorption[~df.YAGFired & df.AbsorptionON].hist(
            bins=20, ax=self.ax[0, 0]
        )
        self.ax[0, 0].axvline(df.attrs["absorption_cutoff"], c="k", ls="--")
        self.ax[0, 0].set_xlabel("Integrated absorption")

    def plot_YAG_firing(self, df: pd.DataFrame) -> None:
        """
        Plot to check if there were "molecule pulses" where the YAG didn't fire
        """
        df.YAGPD.apply(np.max).plot.line(ax=self.ax[0, 1])
        self.ax[0, 1].axhline(df.attrs["YAG_cutoff"], c="k", ls="--")
        self.ax[0, 1].set_xlabel("Data row")
        self.ax[0, 1].set_ylabel("YAG PD signal")

    def plot_absorption_ON(self, df: pd.DataFrame) -> None:
        """
        Plot to check that absorption laser was on
        """
        df["AbsNormPD"].apply(np.min).plot.line(ax=self.ax[1, 0])
        self.ax[1, 0].axhline(df.attrs["absorption_on_cutoff"], c="k", ls="--")
        self.ax[1, 0].set_xlabel("Data row")
        self.ax[1, 0].set_ylabel("Absorption norm PD signal")

    def plot_RC_ON(self, df: pd.DataFrame) -> None:
        """
        Plot to check that RC laser was on
        """
        df["RCPD"].apply(np.min).plot.line(ax=self.ax[1, 1])
        self.ax[1, 1].axhline(df.attrs["RC_on_cutoff"], c="k", ls="--")
        self.ax[1, 1].set_xlabel("Data row")
        self.ax[1, 1].set_ylabel("RC PD signal")


@dataclass
class GaussianFitPlotter(PlotterDefault):
    """
    Used for plotting a fluorescence image and a gaussian fit to it.
    """

    save: bool = False

    def plot(self, image: Image, result: lmfit.model.ModelResult) -> None:
        # Get parameters needed for plotting
        self.get_params(image.values, result)

        # Make the plot
        self.setup_plots()
        self.plot_image(image.values)
        self.plot_fit(result)
        # self.add_colorbar()
        self.plot_fit_maximum(result)
        self.plot_cut_along_vertical(image.values)
        self.plot_vline()
        self.plot_cut_along_horizontal(image.values)
        self.plot_hline()
        self.plot_fit_x(result)
        self.plot_fit_y(result)
        if image.scan_param or image.switch:
            self.add_title(image.scan_param, image.switch)
            if self.save:
                self.save_plot()

        plt.show()

    def get_params(self, image: np.ndarray, result: lmfit.model.ModelResult) -> None:
        """
        Gets some parameters needed for plotting from the provided image and fit result
        """
        self.image_shape = image.shape  # Dimensions of the image
        self.center_x = result.params["center_x"].value
        self.center_y = result.params["center_y"].value

    def setup_plots(self) -> None:
        """
        Sets up figure and axes
        """
        self.fig, self.ax = plt.subplots(figsize=(16, 9))

        divider = make_axes_locatable(self.ax)
        self.top_ax = divider.append_axes("top", 1.05, pad=0.1, sharex=self.ax)
        self.right_ax = divider.append_axes("right", 1.05, pad=0.1, sharey=self.ax)

        # Get rid of some ticklabels
        self.top_ax.xaxis.set_tick_params(labelbottom=False)
        self.right_ax.yaxis.set_tick_params(labelleft=False)

    def plot_image(self, image: np.ndarray) -> None:
        """
        Plots the fluorescence image
        """
        self.imag = self.ax.imshow(image, origin="lower")
        self.ax.autoscale(enable=False)

    def plot_fit(self, result: lmfit.model.ModelResult) -> None:
        """
        Plots the fit to the image
        """
        # Find the ranges of the x and y axes
        x_range = np.arange(0, self.image_shape[0])
        y_range = np.arange(0, self.image_shape[1])

        # Make meshgrid out of the axes
        X, Y = np.meshgrid(x_range, y_range)

        # Flatten the mehgrid arrays to get x and y coordinates for flattened image
        x_plot, y_plot = X.flatten(), Y.flatten()

        self.ax.contour(
            result.eval(x=x_plot, y=y_plot).reshape(self.image_shape), origin="lower"
        )

    def plot_fit_x(self, result: lmfit.model.ModelResult) -> None:
        """
        Plots fit along the x direction
        """
        x_range = np.arange(0, self.image_shape[0])
        self.top_ax.plot(result.eval(x=x_range, y=self.center_y))

    def plot_fit_y(self, result: lmfit.model.ModelResult) -> None:
        """
        Plots fit along the x direction
        """
        y_range = np.arange(0, self.image_shape[1])
        self.right_ax.plot(result.eval(x=self.center_x, y=y_range), y_range)

    def add_colorbar(self) -> None:
        """
        Adds a colorbar to the plot
        """
        self.fig.colorbar(self.imag, ax=self.ax, shrink=0.9)

    def plot_fit_maximum(self, result: lmfit.model.ModelResult) -> None:
        """
        Show the center of the Gaussian on the plot
        """
        # Plot an x at the maximum
        self.ax.plot(
            self.center_x, self.center_y, color="red", marker="x", markersize=10
        )

    def plot_cut_along_vertical(self, image: np.ndarray) -> None:
        """
        Plot the data along a vertical line through the maximum
        """
        self.right_ax.plot(
            image[:, int(self.center_x)], np.arange(0, self.image_shape[1])
        )

    def plot_vline(self) -> None:
        """
        Show a vertical line on the plot to show where the vertical cut is placed
        """
        self.ax.axvline(self.center_x, color="k", ls="--")

    def plot_cut_along_horizontal(self, image: np.ndarray) -> None:
        """
        Plot the data along a vertical line through the maximum
        """
        self.top_ax.plot(
            np.arange(0, self.image_shape[0]), image[int(self.center_y), :]
        )

    def plot_hline(self) -> None:
        """
        Show a vertical line on the plot to show where the vertical cut is placed
        """
        self.ax.axhline(self.center_y, color="k", ls="--")

    def add_title(self, scan_param: ScanParam, switch: Switch) -> None:
        """
        Adds a title to the plots
        """
        title = ""
        if scan_param:
            title += f"{scan_param.name} = {scan_param.value}"

        if switch:
            title += f", {switch.name} = {switch.value}"

        self.fig.suptitle(title)

    def save_plot(self):
        save_dict = {"fig": self.fig, "ax": [self.ax, self.top_ax, self.right_ax]}
        with open(
            PLOT_DIR / Path(self.fig._suptitle.get_text() + ".pickle"), "wb+"
        ) as f:
            pickle.dump(save_dict, f)


@dataclass
class ParamScanPlotter(Plotter):
    """
    Plots the results from a parameter scan
    """

    figsize: Tuple[int] = (16, 9)

    def plot(self, df: pd.DataFrame, param_name: str, signal_name: str) -> None:
        fig, ax = plt.subplots(figsize=(16, 9))
        ax.errorbar(df[param_name], df[signal_name], marker="x")
        ax.set_xlabel(param_name, fontsize=16)
        ax.set_ylabel(signal_name, fontsize=16)

        plt.show()


@dataclass
class SwitchingParamScanPlotter(Plotter):
    """
    Plots the results from a parameter scan
    """

    figsize: Tuple[int] = (16, 9)

    def plot(
        self, df: pd.DataFrame, param_name: str, signal_name: str, switch_name: str
    ) -> None:
        fig, ax = plt.subplots(figsize=(16, 9))
        ax.errorbar(
            df[param_name], df[signal_name + "_ON"], marker="x", label=switch_name
        )
        ax.errorbar(
            df[param_name],
            df[signal_name + "_OFF"],
            marker="x",
            label="NOT " + switch_name,
        )
        ax.set_xlabel(param_name, fontsize=16)
        ax.set_ylabel(signal_name, fontsize=16)
        ax.legend()

        plt.show()


@dataclass
class ParamScanPlotterBS(Plotter):
    """
    Plots the results from a parameter scan
    """

    figsize: Tuple[int] = (16, 9)

    def plot(self, df: pd.DataFrame, param_name: str, signal_name: str) -> None:
        fig, ax = plt.subplots(figsize=(16, 9))
        ax.errorbar(
            df[param_name], df[signal_name], yerr=df[signal_name + "_err"], marker="x"
        )
        ax.set_xlabel(param_name, fontsize=16)
        ax.set_ylabel(signal_name, fontsize=16)

        plt.show()


@dataclass
class SwitchingParamScanPlotterBS(Plotter):
    """
    Plots the results from a parameter scan
    """

    switch_name: str
    figsize: Tuple[int] = (16, 9)

    def plot(self, df: pd.DataFrame, param_name: str, signal_name: str) -> None:
        fig, ax = plt.subplots(figsize=(16, 9))
        ax.errorbar(
            df[param_name],
            df[signal_name + "_ON"],
            yerr=df[signal_name + "_ON_err"],
            marker="x",
            label=self.switch_name,
        )
        ax.errorbar(
            df[param_name],
            df[signal_name + "_OFF"],
            yerr=df[signal_name + "_OFF_err"],
            marker="x",
            label="NOT " + self.switch_name,
        )
        ax.set_xlabel(param_name, fontsize=16)
        ax.set_ylabel(signal_name, fontsize=16)
        ax.legend()
        plt.show()
