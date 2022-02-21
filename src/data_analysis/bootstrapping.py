"""
Bootstrapper objcects that are used to get errorbars for fit results
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import median_abs_deviation
from tqdm import tqdm

from .analyzers import Analyzer
from .plotters import Plotter


def mad_err(series: pd.Series) -> float:
    """
    Estimates uncertainties based on median absolute deviation from median. Scaled
    to corresponed to standard error of gaussian if series is normally distributed
    """
    return median_abs_deviation(series, scale="normal")


class Bootstrapper:
    def __init__(
        self,
        analyzer: Analyzer,
        central_func=np.median,
        error_func=mad_err,
        plotter: Plotter = None,
    ) -> None:
        self.analyzer = analyzer
        self.central_func = central_func
        self.error_func = error_func
        self.plotter = plotter

        # Initialize containers for results
        self.df_bootstrap = pd.DataFrame()
        self.df_agg = pd.DataFrame()

    def bootstrap(self, df: pd.DataFrame, n_bs=3, n_jobs=8) -> None:
        """
        Bootstraps the analysis performed by analysis function by repeating the analysis multiple
        times with subsets of data (with replacement)
        """
        # Get analysis function from analyzer
        analysis_function = self.analyzer.analyze_data

        # Container for results
        df_bootstrap = pd.DataFrame()

        # Perform the bootstrap in parallel
        df_bootstrap = Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(analysis_function)(df.sample(replace=True, frac=0.5))
            for _ in range(n_bs)
        )
        self.df_bootstrap = pd.concat(df_bootstrap, ignore_index=True)

    def aggregate(self, scan_param=None):
        """
        Aggregates the bootstrapped results.
        """
        # If scan parameter provided, group data by scan param before aggregation. Else, aggregate
        # on all data
        if scan_param:
            df_agg = self.df_bootstrap.groupby(by=scan_param).agg(
                [self.central_func, self.error_func]
            )
        else:
            df_agg = self.df_bootstrap.agg([self.central_func, self.error_func])

        # Rename columns
        new_columns = [
            col[0] + "_err" if col[1] == self.error_func.__name__ else col[0]
            for col in df_agg.columns
        ]

        df_agg.columns = new_columns

        self.df_agg = df_agg.reset_index()

        if self.plotter:
            for analyzer in self.analyzer.analyzers:
                self.plotter.plot(
                    self.df_agg,
                    self.analyzer.scan_param,
                    analyzer.signal_calculator.signal_name,
                )
