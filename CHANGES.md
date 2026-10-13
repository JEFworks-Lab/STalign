# Changes

## Unreleased

This version keeps every function and argument of earlier versions. Alignments run much faster, and several bugs are fixed, so results are close to, but not identical with, earlier results.

### Faster

- `LDDMM` and `LDDMM_3D_to_slice` have a new optional C++ engine, compiled by `pip` when a C++17 compiler is available. Without one, they use a new pure `torch` implementation. Both compute the same iterations up to rounding. Speed per iteration on an Apple M1 Ultra (16 threads) against the previous version, with the resulting time for the tutorial's fit:

  | | C++ engine | `torch` engine |
  |---|---|---|
  | 2D, MERFISH tutorial (10,000 iterations: ~14 min before) | 81–91x (~11 s) | ~10x (~1.6 min) |
  | 3D, MERFISH-to-Allen tutorial (2,000 iterations: ~22 min before) | 39–45x (~35 s) | ~10x (~2.5 min) |

- `LDDMM` and `LDDMM_3D_to_slice` draw their figures once, after the last iteration, instead of every 10 iterations (`draw=True` restores the old behavior).
- `rasterize` is about 50x faster (86,000 cells: 2 s to 35 ms; 2 million cells: 47 s to 0.85 s), and `rasterize_with_signal` about 60x with one signal column and about 20x with 16 columns. Both give the same images up to rounding (1e-12 relative). Peak memory is bounded.
- The transform functions split large point sets into chunks and use several threads.

### New options

- `LDDMM` and `LDDMM_3D_to_slice` take keyword-only arguments `backend` (`'auto'`, `'cpp'` or `'torch'`), `num_threads` and `draw`. The new `backend_info()` reports which engine is used and, if the C++ engine is unavailable, why.
- The number of threads is at most `num_threads`, else the environment variable `STALIGN_NUM_THREADS`, else `torch.get_num_threads()`. Results do not depend on it.
- `transform_points_source_to_target`, `build_transform(direction='f')`, `build_transform3D(direction='f')` and `transform_image_target_to_source` take `exact_inverse` (default `True`, see below).

### Changes that affect results

- **Source-to-target transforms use the exact inverse of the fitted mapping.** `LDDMM` fits a target-to-source map; mapping cells the other way previously used a forward Euler approximation that is not its inverse. On the tutorial fits, cells move by 1–13 µm on average and up to ~100 µm. The ground-truth error of the simulation tutorial drops from 9.68 to 6.68 µm. `exact_inverse=False` gives the previous approximation.
- **`transform_points_target_to_source` integrates the velocity in the correct time order** (t = nt-1 down to 0, as `LDDMM` and `build_transform` do). Changes are small: at most 0.26 raster pixels on the tutorials.
- **The 3D regularizer is corrected.** `LDDMM_3D_to_slice` transformed the velocity over only two of its three axes, so it applied no smoothness left-to-right and a position-dependent penalty up to 83,521 times stronger at the midline. It now computes the documented 0.5/sigmaR² Σ_t ‖L v_t‖². At the current default `sigmaR=1e8` the corrected term is weak; a new 3D default is being evaluated.
- **`analyze3Dalign` returns correct atlas coordinates.** It indexed a 10-unit grid with indices computed from `dx`, so coordinates were wrong whenever `dx` was not 10 (by 4.3 mm on average in the STARmap tutorial, `dx=50`) and biased by up to 19 µm otherwise. Region labels were mostly right and change for under 3% of cells at region boundaries.
- `LDDMM` and `LDDMM_3D_to_slice` return the affine `A`, velocity `v` and (3D) sample points `Xs` of the same, last evaluated iteration. `v` previously came from one update later than `A`. The difference is one velocity step.
- `muA` and `muB` are matched to the number of target channels. A 3-entry mean against a 1-channel target (as in the 3D tutorials) previously counted the squared distance three times; it now warns and uses the single value. Integer, float, list, numpy and torch values are accepted.
- `rasterize_with_signal` keeps the sign of a signed signal (it returned the absolute value). `normalize` maps a constant array to `t_min` instead of NaN, so `rasterize` with a constant gene gives a zero image instead of NaN.

Long 3D fits are very sensitive to rounding: a change in the last digit of the input moves the original code's own result by 30–90 µm on average. Small differences from earlier versions are therefore expected even where nothing was fixed.

### Bug fixes

- `rasterizePCA` works (it raised `NameError: name 'tools' is not defined`) and tolerates genes without counts.
- `build_transform3D(direction='f')` works (it always raised an error) and accepts `v=None`.
- `v_to_phii` and `v_to_phii_3D` accept the velocity layout that `LDDMM` returns.
- `calculate_tre` and the transform functions accept numpy arrays, torch tensors and lists of integers or floats.
- `point_annotator.py` and `curve_annotator.py` work with output file names (they raised `NameError`).
- `saveRasters` writes into the folder `path` (it wrote next to it without a trailing slash).
- `download_aba_ontology` and `download_aba_image_labels` raise on HTTP errors instead of saving the error page.
- `plot_brain_regions` gives every region its own color with a readable legend; both plot helpers return `(fig, ax)`.
- `LDDMM_3D_to_slice` warns that it ignores `pointsI` and `pointsJ` (it has no landmark term).
- Inputs are checked before fitting, with error messages that name the problem. A drawing error warns instead of stopping the fit.

### Packaging

- `requirements.txt` lists only the runtime dependencies, unpinned: matplotlib, numpy, pandas, pynrrd, requests and torch. pandas, pynrrd and requests are needed only by the Allen atlas helpers. The tutorials also use plotly, scipy, scikit-learn and scikit-image.
- `pyproject.toml` declares the build requirements. pybind11 is needed only to compile the C++ engine; if compilation fails, installation still succeeds and the `torch` engine is used.
- A `pytest` suite in `tests/` checks both engines against a plain-autograd reference implementation of the iteration (`tests/reference.py`), and the rasterization and transforms against the previous version.
