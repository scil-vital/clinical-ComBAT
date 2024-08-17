#!/usr/bin/env python
# -*- coding: utf-8 -*-
from glob import glob
from os import path

from setuptools import setup

here = path.abspath(path.dirname(__file__))

with open("README.md", "r") as fh:
    long_description = fh.read()
#TODO github et nom package
setup(
    author="Gabriel Girard, Manon Edde, Yoan David, Félix Dumais, Matthieu Dumont, Guillaume Theaud, Jean-Christophe Houde, Maxime Descoteaux, Pierre-Marc Jodoin",
    author_email="gabriel.p.girard@usherbrooke.ca",
    classifiers=[        
        "Programming Language :: Python :: 3.9",
    ],
    description="ComBAT harmonization algorithm.",

    url="https://github.com/jodoin/clinical-ComBAT",
    project_urls={
        "Github": "https://github.com/jodoin/clinical-ComBAT",
    },
    name="clinical_combat",
    packages=["clinical_combat"],
    version="1.0.0",
    zip_safe=False,
    scripts=glob("scripts/*.py"),
)
