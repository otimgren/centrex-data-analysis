"""
Code for performing data analysis for State Preparation A tests that were performed 
using the EMCCD camera in late 2021.
"""
from pathlib import Path

import lmfit
import matplotlib.pyplot as plt
import numpy as np
from data_analysis import analyzers
from data_analysis import background_subtractors as BG_subtract
from data_analysis import (
    bootstrapping,
    cutters,
    plotters,
    preprocessors,
    retrievers,
    signal_calculators,
)


def main():
    ##### Retrieving data from file #####
    # Initialize data retriever
    SPA_retriever = retrievers.SPARetriever()

    # Define path to data
    DATA_DIR = Path(
        "F:\Google Drive\CeNTREX Oskari\State preparation"
        "\From rotational cooling to ES lens\Data analysis\Data"
    )
    DATA_FNAME = Path("SPA_test_11_9_2021.hdf")
    filepath = DATA_DIR / DATA_FNAME

    # Print datasets in data
    SPA_retriever.print_run_names(filepath)

    # Retrieve data
    scan_param_name = "SPAJ01Frequency"
    df = SPA_retriever.retrieve_data(
        filepath,
        6,
        scan_param="SynthHD Pro SPA SetFrequencyCHAGUI",
        scan_param_new_name=scan_param_name,
    )

    ##### Processing data #####
    # Define preprocessors
    processors = [
        preprocessors.NormalizedAbsorption(),
        preprocessors.IntegratedAbsorption(),
        preprocessors.AbsorptionBigEnough(),
        preprocessors.YAGFired(),
        preprocessors.AbsorptionON(),
        preprocessors.RotCoolON(),
        preprocessors.RCShutterOpen(),
        preprocessors.CamDAQTimeDiff(),
        preprocessors.MicrowavesON(),
    ]

    # Define plotters that will be run after the preprocessing
    plotters_list = [plotters.PreProcessorPlotter()]

    # Define the data processing pipeline
    processor_pipeline = preprocessors.ProcessorPipeline(
        processors, plotters=plotters_list
    )

    # Pre-process the data
    processor_pipeline.process_data(df, plot=False)

    ##### Cutting data #####
    # Define cutters
    cutters_list = [
        cutters.YAGFiredCutter(),
        cutters.AbsorptionONCutter(),
        cutters.AbsBigEnoughCutter(),
        cutters.TimingCutter(),
    ]

    # Define cutter pipeline
    cutter_pipeline = cutters.CutterPipeline(cutters_list)

    # Apply cuts
    cutter_pipeline.apply_cuts(df, print_result=True)

    ##### Analyze preprocessed data #####
    # Define a background subtractor
    df_background = SPA_retriever.retrieve_data(filepath, 0)
    background_subtractor = BG_subtract.AcquiredBackgroundSubtractor(df_background)

    # Define a signal size calculator
    init_params = lmfit.Parameters()
    init_params.add("A", value=5, min=0)
    init_params.add("center_x", value=200, min=0, max=512)  # , vary = False)
    init_params.add("center_y", value=250, min=0, max=512)  # , vary = False)
    init_params.add("phi", value=0, min=0, max=np.pi / 4)
    init_params.add("sigma_x", value=16, min=10, max=100)
    init_params.add("sigma_y", value=30, min=10, max=100)
    init_params.add("C", value=0)
    signal_size_calculator = signal_calculators.SignalFromGaussianFit(  # plotter = GaussianFitPlotter(),
        # init_params = init_params,
        ROI=np.s_[150:450, 100:300]
    )

    # Define an analyzer
    analyzers_list = [
        analyzers.FluorescenceImageAnalyzer(
            background_subtractor, signal_size_calculator
        )
    ]

    # Define a parameter scan analyzer
    switch_name = "MicrowavesON"
    scan_analyzer = analyzers.SwitchingParamScanAnalyzer(
        scan_param_name,
        switch_name,
        analyzers_list,
        # plotter = SwitchingParamScanPlotter()
    )

    # Run parameter scan analysis using bootstrap
    bootstrapper = bootstrapping.Bootstrapper(
        scan_analyzer, plotter=plotters.SwitchingParamScanPlotterBS()
    )
    bootstrapper.bootstrap(df, n_bs=10)
    df_bootstrap = bootstrapper.df_bootstrap
    bootstrapper.aggregate(scan_param=scan_param_name)
    df_agg = bootstrapper.df_agg


if __name__ == "__main__":
    main()
