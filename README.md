# Important note
This is a work in progress!

### 3D atlas to merfish

The notebook which was used for allen to merfish alignment, more_slices_v01_loop.ipynb.

This requires csv files describing cell locations (*metadata*.csv).

It requires the allen atlas nissl image and annotation (http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/ara_nissl/ara_nissl_50.nrrd, http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/annotation_50.nrrd).

The notebooks with higher version numbers are trying to estimate the contrast of each structure individually. We never finalized this.

### 2D Merfish to atlas
Osagie uploaded standardized notebook called 'merfish_atlas_tools_v00.'


### Merfish to visium
For merfish to visium alignment, the latest notebook to use is merfish_visium_v02_tweak_params.ipynb

We load and rasterize the data.

Skip over In 17 (it has been replaced).

Registration starts on In 27.  Kalen will try to reproduce this for Merfish to Merfish.  She will have to rasterize two Merfish datasets.

Osagie uploaded standardized notebook for this application called 'merfish_visium_tools_v02.' Additionally, we still need to optimaize parameters of LDDMM algorithm.

### Merfish to Merfish
Osagie uploaded standardized notebook called 'merfish_merfish_tools_v01.'


# Spatial Transcriptomics Registration
This repository contains software tools and examples for registering pairs of spatial transcriptomics datasets to each other, or single datasets to an atlas.

The package by converting datasets to cell density images, and using LDDMM based registration tools.  We include automatic and semiautomatic (with landmark placement) tools.

Two examples are included, illustrating registration of MERFISH to Visium data, and Allen Common Coordinate atlas to MERFISH data.


Here show example figure, same one Jean is workin on.


## 3D to 2D atlas to slice registration


## Interactive landmarking tool

## Experiments involving gene expression data
This can be found in the following notebooks:
- TranscriptomicsDataVisualization.ipynb
- rasterizeSlices.ipynb
- registration_pca.ipynb
- accuracy_tests_noPoints.ipynb
- accuracy_figures_noPoints.ipynb
