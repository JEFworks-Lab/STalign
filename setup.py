from setuptools import setup
from setuptools import find_packages

with open('requirements.txt') as f:
	required = f.read().splitlines()

setup(
    name='STalign',
    author='JEFworks',
    version='1.0',
    include_package_data=True,
    description='aligning spatial transcriptomics data',
    packages=find_packages(),
    install_requires = required
)
