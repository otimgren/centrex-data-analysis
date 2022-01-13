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
    install_requires=["numpy", "pandas", "joblib"],
)
