# centrex-data-analysis
Code used for analyzing data for the CeNTREX experiment.

## Getting started
- I suggest first creating a clean virtual environment, e.g. using conda by running `conda create --name [environment name] python==3.10` (might also work with newer or older versions of Python).
- Then run `python setup.py install` in the root folder of the repository to install the data analysis package and its dependencies.
- You should then be able to run the scripts in the folders for each part of the experiment (assuming you have the data available and sort out the paths to the data).

## Flow of data
<img width="789" alt="image" src="https://user-images.githubusercontent.com/34794187/150867757-e6734047-72f5-4826-9977-6787e1750be2.png">

1. Retrieve data from storage (e.g. an hdf-file) into a `pandas.DataFrame` using a `retrievers.Retriever`
2. Preprocess the data using `preprocessors.PreProcessor`. This could be e.g. calculating absorption signals, setting tags for cuts etc. **NOTE**: preprocessing should only use data for a single row (i.e. molecule pulse) at a time since undesirable data has not yet been cut, and aggregating data at this point would thus include that undesirable data also. A `preprocessors.ProcessorPipeline` can be used to combine multiple `PreProcessor`s together.
3. Cut undesirable data using `cutters.Cutter` objects based on tags set by the `preprocessors`. Reasons to cut could be e.g. YAG not firing, absorption signal too small, laser turning off etc. Again, a `cutters.CutterPipeline` can be used to combine multiple `Cutter`s together.
4. Analyze data using `analyzers.Analyzer` objects. This part involves calculating the "signal" (usually integrated fluorescence in a camera image or PMT trace) and aggregating the data based on some scanned parameter using `ParamScanAnalyzer` (e.g. laser frequency) and possibly a switch using `SwitchingParamScanAnalyzer` (e.g. rotational cooling laser  switched on and off while frequency scanned at the same time). After this we have the final product: a `pandas.DataFrame` of analyzed data.
5. Save the analyzed data to file using standard `pandas` methods, e.g. `pandas.DataFrame.to_hdf` (can be read using `pandas.read_hdf`).

## Analysis of images
This section details how camera images of fluorescence are analyzed, as there are some special considerations that go into this. The `analyzers.Analyzer` that is used with camera images is `analyzers.FluorescenceImageAnalyzer`.

### Background subtraction
The camera images suffer from a random background that can subtracted from the images to improve the signal due to the fluorescing molecules. There are multiple ways to do this, which why the `FluorescenceImageAnalyzer` is initialized with a `background_subtractors.BackgroundSubtractor` which takes care of the background subtraction in whatever way is desirable, e.g. by subtracting an acquired background, or based on a fit to the image around the part where the fluorescence signal is.

### Signal calculation
Similar to background subtraction, there are multiple ways to calculate the signal size for camera images, and the `FluorescenceImageAnalyzer` is thus also initialized with a `signal_calculators.SignalCalculator`. The `SignalCalculator` takes care of calculating the signal in the desired way, e.g. by fittign a 2D Gaussian to the image, or by summing the counts in each pixel of the image.

## Bootstrapping
Bootstrapping can be useful to estimate uncertainties for the signal size in some scenarios. For example, when analyzing camera image data, a fit is used to calculate the size of the signal. The fitting procedure doesn't give reliable errorbars, so bootstrapping is used instead. The `bootstrapping.Bootstrapper` class can be used to run the `Analyzer`s (from step 4. of "Flow of Data") multiple times with different subsamples of the data. The distribution of the results for different subsamples is then used to estimate the uncertainties in the results.
