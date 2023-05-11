from setuptools import find_packages, setup
from Cython.Build import cythonize

setup(
    name="pyfx",
    version="0.1.0",
    author="Calvin Leung",
    author_email="calvinleung@mit.edu",
    packages=['pyfx'],
    package_dir = {
        '':"src"
    },
    url="http://github.com/leungcalvin/pyfx",
    license="LICENSE.txt",
    description="A Python and HDF5-based VLBI correlator for widefield, transient VLBI",
    long_description=open("README.md").read(),
    install_requires=[
        "numpy >= 1.19.2",
        "scipy >= 1.5.2",
        "matplotlib",
        "h5py",
        "argparse",
        "astropy",
        "baseband-analysis",
        "difxcalc-wrapper",
    ],
    python_requires=">=3.6",
)
