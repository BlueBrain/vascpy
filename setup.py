#!/usr/bin/env python
from setuptools import find_packages, setup


DEPS_VTK = ["vtk>=8.1.2"]


setup(
    classifiers=[
        "Programming Language :: Python :: 3.8",
    ],
    name="vascpy",
    description="Vasculature API",
    author="Blue Brain Project, EPFL",
    install_requires=[
        "numpy>=1.17",
        "scipy>=1.0.0",
        "h5py>=3.5.0",
        "pandas>=1.0.0",
        "morphio>=3.0.0",
        "libsonata>=0.1.8",
        "click>=7.0",
    ],
    extras_require={"convert-vtk": DEPS_VTK, "all": DEPS_VTK + ["python-igraph>=0.8.0"]},
    packages=find_packages(),
    entry_points={"console_scripts": ["vascpy = vascpy.cli:app"]},
    include_package_data=True,
    use_scm_version=True,
    python_requires=">=3.7",
)
