from setuptools import find_packages, setup

VERSION = "0.1"
DESCRIPTION = "Data analysis tools for the CeNTREX experiment."

setup(
    name="data_analysis",
    version=VERSION,
    author="Oskari Timgren",
    description=DESCRIPTION,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.22",
        "pandas>=1.4",
        "joblib>=1.1",
        "lmfit>=1.0.3",
        "matplotlib>=3.5",
        "tqdm>=4.62",
        "h5py>=3.6",
        "pytables>=3.6.1",
    ],
)
