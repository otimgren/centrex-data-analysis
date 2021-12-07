from pathlib import Path

from matplotlib.pyplot import plot

from data_analysis.cutters import (CutterPipeline, YAGFiredCutter, AbsorptionONCutter,
                                   AbsBigEnoughCutter, TimingCutter, RotCoolOFFShutterOpenCutter)
from data_analysis.plotters import PreProcessorPlotter
from data_analysis.preprocessors import (AbsorptionON, ProcessorPipeline, NormalizedAbsorption, 
                                      IntegratedAbsorption, YAGFired, AbsorptionBigEnough,
                                      RCShutterOpen, RotCoolON, CamDAQTimeDiff)
from data_analysis.retrievers import SPARetriever

def main():
    ##### Retrieving data from file #####
    # Initialize data retriever
    SPA_retriever = SPARetriever()

    # Define path to data
    DATA_DIR =  Path("F:\Google Drive\CeNTREX Oskari\State preparation"
                     +"\From rotational cooling to ES lens\Data analysis\Data")
    DATA_FNAME = Path("SPA_test_11_4_2021.hdf")
    filepath = DATA_DIR / DATA_FNAME

    # Print datasets in data
    SPA_retriever.print_run_names(filepath)

    # Retrieve data
    df = SPA_retriever.retrieve_data(filepath, 3)
    
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
    ]

    # Define plotters that will be run after the preprocessing
    plotters = [
        PreProcessorPlotter()
    ]

    # Define the data processing pipeline
    processor_pipeline = ProcessorPipeline(processors, plotters=plotters)
    print(processor_pipeline)

    # Pre-process the data
    processor_pipeline.process_data(df, plot = True)

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

    print(df.head().IntegratedAbsorption)

if __name__ == "__main__":
    main()