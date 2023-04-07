# STalign Overview
Python tool for alignment of spatial transcriptomics (ST) data using diffeomorphic metric mapping.

[fig 1]

[bioRxiv link]

## Overview

STalign enables:
- alignment of single-cell resolution ST datasets within technologies
- alignment single-cell resolution ST datasets to histology
- alignment of ST datasets across technologies
- alignment of ST datasets to a 3D common coordinate framework 

## Installation
`pip install "git+https://github.com/JEFworks-Lab/STalign.git"`

### Input Data
To use this tool, you will need provide the following information:

2D-2D alignment:
- Source Image: Arrays of x and y positions of cells
- Target Image: Arrays of x and y positions of cells
3D-2D alignment:
- Source Image: (Default: Adult mouse Allen Brain Altas CCFv3) 3D Matrix with voxels corresponding to (1) cell intensity and (2) brain regions
- Target Image: Arrays of x and y positions of cells

## Usage

To use `STalign`, please refer to the following Jupyter Notebooks with usage examples:

[Xenium-Xenium Alignment](https://jef.works/STalign/notebooks/xenium-xenium-alignment.html) <br />
[Xenium-H&E Image Alignment](https://jef.works/STalign/notebooks/xenium-heimage-alignment.html) <br />
[3D alignment to the Allen Brain Atlas](https://jef.works/STalign/notebooks/merfish-allen3Datlas-alignment.html) <br />

