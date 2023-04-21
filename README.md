# STalign

`STalign` aligns spatial transcriptomics (ST) tissue sections to each other and to 3D atlases like the Allen Brain Atlas using algorithms that build upon diffeomorphic metric mapping. 

More information regarding the overall approach, methods and validations can be found on the bioRxiv preprint:
<a href="https://www.biorxiv.org/content/10.1101/2023.04.11.534630v1">
<b>Alignment of spatial transcriptomics data using diffeomorphic metric mapping</b>
Kalen Clifton*, Manjari Anant*, Osagie K Aimiuwu, Justus M Kebschull, Michael I Miller, Daniel Tward^, Jean Fan^
</a>

## Overview

<img src="https://jef.works/assets/papers/STalign_anim_small.gif">

STalign enables:
- alignment of single-cell resolution ST datasets within technologies
- alignment single-cell resolution ST datasets to histology
- alignment of ST datasets across technologies
- alignment of ST datasets to a 3D common coordinate framework 

## Installation
`pip install "git+https://github.com/JEFworks-Lab/STalign.git"`

## Input Data
To use this tool, you will need provide the following information:

### single-cell resolution ST alignment
- Source: Arrays of x and y positions of cells
- Target: Arrays of x and y positions of cells

### single-cell and spot-resolution ST alignment
- Source: Arrays of x and y positions of cells from single-cell resolution ST data
- Target: Registered H&E image from spot-resolution ST data

### 3D-2D alignment
- Source: (Default: Adult mouse Allen Brain Altas CCFv3) 3D Matrix with voxels corresponding to (1) cell intensity and (2) annotated tissue regions
- Target: Arrays of x and y positions of cells

## Usage

To use `STalign`, please refer to our [tutorials](jef.works/STalign/tutorials.html) with usage examples.

## Tutorials
To use `STalign`, please refer to the following Jupyter Notebooks with usage examples: <br />
- [Xenium-Xenium Alignment](https://jef.works/STalign/notebooks/xenium-xenium-alignment.html) <br />
- [Xenium-H&E Image Alignment](https://jef.works/STalign/notebooks/xenium-heimage-alignment.html) <br />
- [3D alignment to the Allen Brain Atlas](https://jef.works/STalign/notebooks/merfish-allen3Datlas-alignment.html) <br />
>>>>>>> 9b545f80abee4a6eeca7a99fcb969dae996cf74a

