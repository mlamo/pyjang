#!/usr/bin/env python
# Filename: setup.py
"""
The jang setup script.

"""
from setuptools import setup


def read_requirements(kind):
    """Return a list of stripped lines from a file"""
    if kind == "install":
        with open("requirements.txt") as fobj:
            return [l.strip() for l in fobj.readlines()]
    else:
        with open(f"requirements-{kind}.txt") as fobj:
            return [l.strip() for l in fobj.readlines()]

try:
    with open("README.md") as fh:
        long_description = fh.read()
except UnicodeDecodeError:
    long_description = "Joint Analysis of Neutrinos and Gravitational waves"

setup(
    name="jang",
    url="https://github.com/mlamo/pyjang/",
    description="Joint Analysis of Neutrinos and Gravitational waves",
    long_description=long_description,
    author="Mathieu Lamoureux",
    author_email="lamoureux.mat@gmail.com",
    packages=["jang"],
    include_package_data=True,
    platforms="any",
    setup_requires=["setuptools_scm"],
    use_scm_version=True,
    python_requires=">=3.6",
    install_requires=read_requirements("install"),
    extras_require={kind: read_requirements(kind) for kind in ["dev"]},
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
    ],
)
