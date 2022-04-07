#!/usr/bin/env python
"""
Copyright (c) 2022 Blue Brain Project/EPFL

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from setuptools import find_packages, setup

DEPS_VTK = ["vtk>=8.1.2"]


setup(
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    name="vascpy",
    description="Vasculature API",
    author="Blue Brain Project, EPFL",
    license="Apache-2",
    url="https://github.com/BlueBrain/vascpy",
    project_urls={
        "Tracker": "https://github.com/BlueBrain/vascpy/issues",
        "Source": "https://github.com/BlueBrain/vascpy",
    },
    use_scm_version={"local_scheme": "no-local-version"},
    setup_requires=["setuptools_scm"],
    install_requires=[
        "numpy>=1.17",
        "scipy>=1.0.0",
        "h5py>=3.4.0",
        "pandas>=1.0.0",
        "morphio>=3.0.0",
        "libsonata>=0.1.8",
        "click>=8.0",
    ],
    extras_require={
        "convert-vtk": DEPS_VTK,
        "all": DEPS_VTK + ["python-igraph>=0.8.0"],
        "docs": ["sphinx", "sphinx-bluebrain-theme"],
    },
    packages=find_packages(),
    entry_points={"console_scripts": ["vascpy = vascpy.cli:app"]},
    include_package_data=True,
    python_requires=">=3.7",
)
