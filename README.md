<img src="https://raw.githubusercontent.com/JEFworks-Lab/STalign/main/STalign_logos_fin.png">

`STalign` aligns spatial transcriptomics (ST) tissue sections to each other and to 3D atlases like the Allen Brain Atlas using algorithms that build upon diffeomorphic metric mapping. 

More information regarding the overall approach, methods and validations can be found in our publication:
<a href="https://www.nature.com/articles/s41467-023-43915-7">
<b>STalign: Alignment of spatial transcriptomics data using diffeomorphic metric mapping</b>
Kalen Clifton*, Manjari Anant*, Gohta Aihara, Lyla Atta, Osagie K Aimiuwu, Justus M Kebschull, Michael I Miller, Daniel Tward^, Jean Fan^
</a>

## Overview

<img src="https://jef.works/assets/papers/STalign_anim_small.gif">

STalign enables:
- alignment of single-cell resolution ST datasets within technologies
- alignment single-cell resolution ST datasets to histology
- alignment of ST datasets across technologies
- alignment of ST datasets to a 3D common coordinate framework 

## Installation & Import

### Installation using pip

This installation method is intended for users who sets up a Python environment without `pipenv`.

```
pip install --upgrade "git+https://github.com/JEFworks-Lab/STalign.git"
```

*All dependencies will be installed into your selected environment with the above command. Dependencies can be found in the requirements.txt file.*

The tutorials also use `plotly`, `scipy`, `scikit-learn` and `scikit-image`:

```
pip install plotly scipy scikit-learn scikit-image
```

### Faster alignment with the C++ engine

`STalign` includes an optional C++ engine for `LDDMM` and `LDDMM_3D_to_slice`. `pip` compiles it during installation if a C++17 compiler is available: the Xcode Command Line Tools on macOS (`xcode-select --install`), `g++` or `clang++` on Linux, or the Microsoft C++ Build Tools on Windows. Without a compiler the installation still succeeds and `STalign` uses its pure `torch` implementation (`pip install -v` shows a warning in the build log). Both compute the same iterations up to rounding (in some long alignments these small differences grow, see [CHANGES.md](CHANGES.md)); the C++ engine is faster. The engine has been tested on macOS (Apple clang) and Linux (gcc 11); Windows builds are expected to work but have not been tested yet.

To check which one is used:

```python
from STalign import STalign
STalign.backend_info()
# {'auto': 'cpp', 'cpp': True, 'cpp_error': None, 'num_threads': 16}
```

`'auto': 'cpp'` means that alignments on the CPU in float64 (the default) use the C++ engine. Otherwise `'cpp_error'` says why it is not available. Alignments on a GPU or in float32 always use `torch`. To choose explicitly, pass `backend='cpp'` or `backend='torch'` to `LDDMM` or `LDDMM_3D_to_slice`.

The alignment uses at most `torch.get_num_threads()` CPU threads. To set a different maximum, set the environment variable `STALIGN_NUM_THREADS` (for example `export STALIGN_NUM_THREADS=8`) or pass `num_threads=8`. The result does not depend on the number of threads.

If you use `STalign` from a clone of this repository without installing it (as the tutorials do with `sys.path.append`), build the C++ engine once, from the base directory of the clone:

```
pip install pybind11
python setup.py build_ext --inplace
```

### Installation using Pipfile from source

This installation method is intended for users who sets up a Python environment with `pipenv`. `pipenv` allows users to create and activate a virtual environment with all dependencies within the Python project. For more information and installation instructions for `pipenv`, see https://pipenv.pypa.io/en/latest/.

Fork and `git clone` the `STalign` github repository.

From the base directory of your local `STalign` git repo, create a `Pipfile.lock` file from `Pipfile` using:

```
pipenv install requests
```

> **_NOTE:_** Since `Pipfile.lock` is platform-dependent and different across operating systems, do not commit `Pipfile.lock` to the git repo if contributing to `STalign` or collaborating with other people.

Activate the virtual environment using:

```
pipenv shell
```

Deactivate the virtual environment using:

```
exit
```

### Import
To import STalign into your Python script, use: 

```python
from STalign import STalign
```

## What changed

This version keeps the functions and arguments of earlier versions, aligns much faster, and fixes several bugs, so alignments are close to, but not identical with, earlier results. For example, the 10,000 iterations of the MERFISH tutorial take about 11 s with the C++ engine and about 1.6 min with `torch`, instead of about 14 min. The changes most likely to matter for existing scripts:

- `LDDMM` and `LDDMM_3D_to_slice` draw their figures once, at the end. `draw=True` redraws them every 10 iterations as before, and `draw=False` turns them off.
- `transform_points_source_to_target` and the other source-to-target transforms use the exact inverse of the fitted transformation; cells can move by up to about 100 µm on the tutorial datasets. `exact_inverse=False` gives the previous approximation.
- The regularization of `LDDMM_3D_to_slice` is corrected and the step size of each affine parameter is controlled (`affine_step_control=True`), so 3D alignments change and no longer depend on rounding. `analyze3Dalign` returns corrected atlas coordinates.
- `LDDMM` and `LDDMM_3D_to_slice` start with a coarse affine pre-alignment that finds large rotations (2D) and tilted slice planes (3D), and choose `sigmaR` and, in 2D, the step sizes from the images (coordinates in µm); `LDDMM` runs 2000 iterations by default. On held-out test problems this more than halved the alignment error on average. Arguments you pass are used as before; `prefit=False` turns the pre-alignment off and `legacy_defaults=True` all of these defaults.

See [CHANGES.md](CHANGES.md) for the full list.

## Input Data
To use this tool, you will need provide the following information:

### single-cell resolution ST alignment
- Source: Arrays of x and y positions of cells
- Target: Arrays of x and y positions of cells

`STalign.align_cells(xI, yI, xJ, yJ)` aligns two such sets of cells in one call: it rasterizes them at 15 µm and runs `LDDMM` coarse to fine, which had about half the error of `LDDMM` on 30 or 60 µm rasters in our tests (see [CHANGES.md](CHANGES.md)). Its result works with the transform functions like that of `LDDMM`.

### single-cell and spot-resolution ST alignment
- Source: Arrays of x and y positions of cells from single-cell resolution ST data
- Target: Registered H&E image from spot-resolution ST data

### 3D-2D alignment
- Source: (Default: Adult mouse Allen Brain Altas CCFv3) 3D Matrix with voxels corresponding to (1) cell intensity and (2) annotated tissue regions
- Target: Arrays of x and y positions of cells

## Usage

To use `STalign`, please refer to our [tutorials](https://jef.works/STalign/tutorials.html) with usage examples.

