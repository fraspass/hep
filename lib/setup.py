#!/usr/bin/env python3
from setuptools import setup

setup(
	name="hep",
	version="0.1",
	packages=[
		"hep",
	],
	install_requires=[
		"numpy",
		"scipy",
		"scikit-learn"
	],
)