from setuptools import find_packages
from setuptools import setup

version = '0.0.1'
description = """A suite containing online gradient descent algorithms
written in python.
"""

install_requires = [
    'numpy',
    'scikit-learn',
    'scipy',
    'matplotlib',
]

tests_require = [
    'mock',
    'pytest',
    'pytest-cov',
]

setup(
    name='ogd',
    version=version,
    description=description,
    author='Benjamin Bossan',
    packages=find_packages(),
    install_requires=install_requires,
    extras_require={'testing': tests_require},
)
