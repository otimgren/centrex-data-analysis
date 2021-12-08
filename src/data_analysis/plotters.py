"""
Classes used for visualizing data
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Rectangle
import lmfit
import numpy as np
import pandas as pd

@dataclass
class Plotter(ABC):
    """
    Abstract parent class for plotters
    """
    figsize: Tuple[int] = (16,9)
    label_fontsize: int = 16
    
    @abstractmethod
    def plot(self, df: pd.DataFrame) -> None:
        """
        Plots the needed plots
        """
        ...

@dataclass
class PreProcessorPlotter(Plotter):
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
        self.fig, self.ax = plt.subplots(self.ncol,self.nrow, figsize = self.figsize)

    def plot_absorption(self, df: pd.DataFrame) -> None:
        """
        Plots the sizes of integrated absorption signals on a histogram
        """
        # Plot for absorption cutoff
        df.IntegratedAbsorption[df.YAGFired & df.AbsorptionON].hist(bins = 20, ax = self.ax[0,0])
        df.IntegratedAbsorption[~df.YAGFired & df.AbsorptionON].hist(bins = 20, ax = self.ax[0,0])
        self.ax[0,0].axvline(df.attrs["absorption_cutoff"], c = 'k', ls = '--')
        self.ax[0,0].set_xlabel('Integrated absorption')

    def plot_YAG_firing(self, df: pd.DataFrame) -> None:
        """
        Plot to check if there were "molecule pulses" where the YAG didn't fire
        """
        df.YAGPD.apply(np.max).plot.line(ax = self.ax[0,1])
        self.ax[0,1].axhline(df.attrs["YAG_cutoff"], c = 'k', ls = '--')
        self.ax[0,1].set_xlabel('Data row')
        self.ax[0,1].set_ylabel('YAG PD signal')

    def plot_absorption_ON(self, df: pd.DataFrame) -> None:
        """
        Plot to check that absorption laser was on
        """
        df["AbsNormPD"].apply(np.min).plot.line(ax = self.ax[1,0])
        self.ax[1,0].axhline(df.attrs["absorption_on_cutoff"], c = 'k', ls = '--')
        self.ax[1,0].set_xlabel('Data row')
        self.ax[1,0].set_ylabel('Absorption norm PD signal')

    def plot_RC_ON(self, df: pd.DataFrame) -> None:
        """
        Plot to check that RC laser was on
        """
        df["RCPD"].apply(np.min).plot.line(ax = self.ax[1,1])
        self.ax[1,1].axhline(df.attrs["RC_on_cutoff"], c = 'k', ls = '--')
        self.ax[1,1].set_xlabel('Data row')
        self.ax[1,1].set_ylabel('RC PD signal')


class GaussianFitPlotter(Plotter):
    """
    Used for plotting a fluorescence image and a gaussian fit to it.
    """
    def plot(self, image: np.ndarray, result: lmfit.model.ModelResult) -> None:
        pass

    def setup_plots(self) -> None:
        """
        Sets up figure and axes
        """
        self.fig, self.ax = plt.subplots(figsize = (32,18))

        divider  = make_axes_locatable(self.ax)
        self.top_ax = divider.append_axes("top", 1.05, pad=0.1, sharex=self.ax)
        self.right_ax = divider.append_axes("right", 1.05, pad=0.1, sharey=self.ax)

        # Get rid of some ticklabels
        self.top_ax.xaxis.set_tick_params(labelbottom=False)
        self.right_ax.yaxis.set_tick_params(labelleft=False)

    def plot_image(self, image: np.ndarray) -> None:
        """
        Plots the fluorescence image
        """
        self.imag = self.ax.imshow(image)
        self.ax.autoscale(enable=False)

    def plot_fit(self, result: lmfit.model.ModelResult) -> None:
        """
        Plots the fit to the image
        """
        self.ax.contour(result.best_fit)

    def add_colorbar(self) -> None:
        """
        Adds a colorbar to the plot
        """
        self.fig.colorbar(self.imag, ax = self.ax, shrink = 0.9)

    def plot_fit_maximum():
        """
        Show the center of the Gaussian on the plot
        """
        





