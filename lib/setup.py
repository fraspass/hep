#!/usr/bin/env python3
from setuptools import setup

setup(
	name="hierarchical_hawkes",
	version="0.1",
	packages=[
		"hierarchical_hawkes",
	],
	install_requires=[
		"numpy",
		"scipy",
		"scikit-learn"
	],
)