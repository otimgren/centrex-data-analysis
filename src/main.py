from pathlib import Path

import lmfit
import matplotlib.pyplot as plt

plt.style.use(["ggplot"])
import numpy as np

from data_analysis.analyzers import (
    FluorescenceImageAnalyzer,
    ParamScanAnalyzer,
    SwitchingParamScanAnalyzer,
)
from data_analysis.background_subtractors import (
    AcquiredBackgroundSubtractor,
    BackgroundSubtractor,
)
from data_analysis.bootstrapping import Bootstrapper
from data_analysis.cutters import (
    AbsBigEnoughCutter,
    AbsorptionONCutter,
    CutterPipeline,
    RotCoolOFFShutterOpenCutter,
    TimingCutter,
    YAGFiredCutter,
)
from data_analysis.plotters import (
    GaussianFitPlotter,
    ParamScanPlotter,
    PreProcessorPlotter,
    SwitchingParamScanPlotter,
    SwitchingParamScanPlotterBS,
)
from data_analysis.preprocessors import (
    AbsorptionBigEnough,
    AbsorptionON,
    CamDAQTimeDiff,
    IntegratedAbsorption,
    MicrowavesON,
    NormalizedAbsorption,
    ProcessorPipeline,
    RCShutterOpen,
    RotCoolON,
    YAGFired,
)
from data_analysis.retrievers import SPARetriever
from data_analysis.signal_calculators import SignalFromGaussianFit


def main():
    ##### Retrieving data from file #####
    # Initialize data retriever
    SPA_retriever = SPARetriever()

    # Define path to data
    DATA_DIR = Path(
        "F:\Google Drive\CeNTREX Oskari\State preparation"
        + "\From rotational cooling to ES lens\Data analysis\Data"
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
        NormalizedAbsorption(),
        IntegratedAbsorption(),
        AbsorptionBigEnough(),
        YAGFired(),
        AbsorptionON(),
        RotCoolON(),
        RCShutterOpen(),
        CamDAQTimeDiff(),
        MicrowavesON(),
    ]

    # Define plotters that will be run after the preprocessing
    plotters = [PreProcessorPlotter()]

    # Define the data processing pipeline
    processor_pipeline = ProcessorPipeline(processors, plotters=plotters)

    # Pre-process the data
    processor_pipeline.process_data(df, plot=False)

    ##### Cutting data #####
    # Define cutters
    cutters = [
        YAGFiredCutter(),
        AbsorptionONCutter(),
        AbsBigEnoughCutter(),
        TimingCutter(),
    ]

    # Define cutter pipeline
    cutter_pipeline = CutterPipeline(cutters)

    # Apply cuts
    cutter_pipeline.apply_cuts(df, print_result=True)

    ##### Analyze preprocessed data #####
    # Define a background subtractor
    df_background = SPA_retriever.retrieve_data(filepath, 0)
    background_subtractor = AcquiredBackgroundSubtractor(df_background)

    # Define a signal size calculator
    init_params = lmfit.Parameters()
    init_params.add("A", value=5, min=0)
    init_params.add("center_x", value=200, min=0, max=512)  # , vary = False)
    init_params.add("center_y", value=250, min=0, max=512)  # , vary = False)
    init_params.add("phi", value=0, min=0, max=np.pi / 4)
    init_params.add("sigma_x", value=16, min=10, max=100)
    init_params.add("sigma_y", value=30, min=10, max=100)
    init_params.add("C", value=0)
    signal_size_calculator = SignalFromGaussianFit(  # plotter = GaussianFitPlotter(),
        # init_params = init_params,
        ROI=np.s_[150:450, 100:300]
    )

    # Define an analyzer
    analyzers = [
        FluorescenceImageAnalyzer(background_subtractor, signal_size_calculator)
    ]

    # Define a parameter scan analyzer
    switch_name = "MicrowavesON"
    scan_analyzer = SwitchingParamScanAnalyzer(
        scan_param_name,
        switch_name,
        analyzers,
        # plotter = SwitchingParamScanPlotter()
    )

    # Run parameter scan analysis using bootstrap
    bootstrapper = Bootstrapper(scan_analyzer, plotter=SwitchingParamScanPlotterBS())
    bootstrapper.bootstrap(df, n_bs=10)
    df_bootstrap = bootstrapper.df_bootstrap
    bootstrapper.aggregate(scan_param=scan_param_name)
    df_agg = bootstrapper.df_agg


if __name__ == "__main__":
    main()
