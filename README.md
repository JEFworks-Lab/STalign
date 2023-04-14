`STalign` aligns spatial transcriptomics (ST) tissue sections to eachother and to 3D atlases like the Allen Brain Atlas using algorithms that build upon diffeomorphic metric mapping. 

More information regarding the overall approach, methods and validations can be found at: [bioRxiv link]

## Overview

STalign enables:
- alignment of single-cell resolution ST datasets within technologies
- alignment single-cell resolution ST datasets to histology
- alignment of ST datasets across technologies
- alignment of ST datasets to a 3D common coordinate framework 

[fig 1]

## Installation
`pip install "git+https://github.com/JEFworks-Lab/STalign.git"`

### Input Data
To use this tool, you will need provide the following information:

#### 2D-2D alignment:
- Source Image: Arrays of x and y positions of cells
- Target Image: Arrays of x and y positions of cells

#### 3D-2D alignment:
- Source Image: (Default: Adult mouse Allen Brain Altas CCFv3) 3D Matrix with voxels corresponding to (1) cell intensity and (2) annotated tissue regions
- Target Image: Arrays of x and y positions of cells

## Usage

To use `STalign`, please refer to our [tutorials](jef.works/STalign/tutorials.html) with usage examples.

