# VessQC: Uncertainty-Guided Curation for 3D Segmentation

[![License BSD-3](https://img.shields.io/pypi/l/VessQC.svg?color=green)](https://github.com/MMV-Lab/VessQC/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/VessQC.svg?color=green)](https://pypi.org/project/VessQC)
[![Python Version](https://img.shields.io/pypi/pyversions/VessQC.svg?color=green)](https://python.org)
[![tests](https://github.com/MMV-Lab/VessQC/workflows/tests/badge.svg)](https://github.com/MMV-Lab/VessQC/actions)
[![codecov](https://codecov.io/gh/MMV-Lab/VessQC/branch/main/graph/badge.svg)](https://codecov.io/gh/MMV-Lab/VessQC)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/VessQC)](https://napari-hub.org/plugins/VessQC)

<!-- 2.1 [File Naming Conventions](#file-naming-conventions)--->
<!-- 2.2 [Generating Uncertainty Maps](#generating-uncertainty-maps)--->
## Table of Contents

1. [Overview](#Overview)
2. [Input Data Requirements](#input-data-requirements)
3. [Launching the Application](#launching-VessQC)
4. [Curation Workflow](#curation-workflow)
5. [Save Behavior](#save-behavior)
6. [Citation](#citation)
7. [Contributing](#contributing)
8. [License](#license)
9. [Issues](#issues)

## Overview
Overview
This repository provides the implementation of **VessQC**, introduced in the manuscript *"Bridging 3D Deep Learning and Uncertainty-Guided Curation for Analysis and High-Quality Segmentation Ground Truth"*, submitted to **ISBI 2026**.

**VessQC** is an open-source, human-in-the-loop tool for efficient, uncertainty-guided curation of large 3D volumetric segmentations, with a particular emphasis on complex vascular structures. By leveraging e.g. pixel-wise or topology-aware uncertainty estimation, VessQC prioritizes regions most likely to contain segmentation errors, thereby significantly improving error recall and reducing manual effort.

This [napari] plugin was generated using [Cookiecutter] and the [@napari] [cookiecutter-napari-plugin] template.

## Input Data Requirements

VessQC requires three core input files per volume, all of which must be perfectly aligned and of identical spatial dimensions (e.g., `.tiff` or `.nii` format):

1. **Raw Image Volume:** The original 3D image data (e.g., lightsheet microscopy stack).  
2. **Segmentation Mask:** A binary segmentation mask of the target structures.  
3. **Uncertainty Map:** A voxel-wise uncertainty map highlighting regions of potential segmentation error.

### File Naming Conventions

VessQC automatically detects corresponding files based on naming patterns. Each file type must follow the conventions below:

| File Type           | Required Suffix   | Example Filename            |
|---------------------|------------------|------------------------------|
| Raw Image Volume    | `_IM`            | `sample01_IM.tiff`           |
| Segmentation Mask   | `_segPred`       | `sample01_segPred.tiff`      |
| Uncertainty Map     | `_uncertainty`   | `sample01_uncertainty.tiff`  |

> Ensure that all files belonging to the same sample share the same prefix (e.g., `sample01`) to allow automatic matching during loading.

### Generating Uncertainty Maps

Uncertainty maps can be generated using models provided in our supplementary repository:
[VessQC-Supplementary](https://github.com/SimPutt/VessQC-Supplementary)

This repository includes both **pixel-wise** and **topology-aware** uncertainty pipelines. Please follow its documentation to produce the required uncertainty maps before running VessQC.

## Installation and Launch

**VessQC** is a plugin that runs within the, open-source image viewer, **napari**. Please follow the steps below to install both napari and the VessQC plugin.

### 1. Install napari (The Host Application)

VessQC requires a working installation of **napari** (Python 3.10-3.13 recommended). For the latest and most detailed instructions, always refer to the official [napari installation guide](https://napari.org/dev/tutorials/fundamentals/quick_start.html).




You can install **VessQC** via [pip]:

```bash
pip install VessQC
```

Or install the latest development version directly from GitHub:

```bash
pip install git+https://github.com/MMV-Lab/VessQC.git
```
Once installation is complete, launch [napari] and start VessQC through the Plugins menu.

## Curation Workflow

The typical workflow involves loading the three required files and then iterating through the following steps:

1. **Load Data:** Use the File menu to load the Image, Segmentation, and Uncertainty Map.

2. **Navigate to High-Uncertainty segments:**  VessQC shows a **ranked list of segmented branches** sorted by their associated uncertainty scores and allows direct **selection of a branch** from the list to automatically crop the area and center to that location

3. **Review and Edit:** Examine the 3D and 2D views at the high-uncertainty location. If an error is identified, use the provided annotation tools to correct the segmentation mask in the 2D viewer.

4. **Iterate:** Continue navigating to the next highest uncertain location and repeat the review and edit process.

5. **Save:** Once the curation process is complete, save the updated segmentation according to your chosen save mode (see below).

## Save Behavior
VessQC offers two save modes to manage the progress and finalization of curated segmentations:

### 1. Temporary Save
* The folder structure remains unchanged.
* Newly curated files are saved with the suffix ```_new```. Example: ```sample01_segPred_new.tiff```
* When the same dataset is reloaded, VessQC automatically prioritizes the ```_new```version for continued editing.

### 2. Final Save
* The finalized image, segmentation, uncertainty map, and curated segmentation are moved to a new ```done/``` subdirectory.

* The moved items are removed from the main directory, ensuring they no longer appear in the list of available images during subsequent sessions.

* This two-tier saving mechanism supports both iterative refinement and systematic completion tracking during large-scale curation projects.

## Citation
If you use the VessQC tool or the methodologies, please cite the corresponding paper:

TODO
<!---
@article{puettmann2025vessqc,
  title={Bridging 3D Deep Learning and Uncertainty-Guided Curation for Analysis and High-Quality Segmentation Ground Truth},
  author={Püttmann, Simon and Sánchez Contreras, Jonathan Jair and Kowitz, Lennart and Lampen, Peter and Gupta, Saumya and Panzeri, Davide and Hagemann, Nina and Xiong, Qiaojie and Hermann, Dirk and Chen, Chao and Chen, Jianxu},
  journal={Proceedings of the IEEE International Symposium on Biomedical Imaging (ISBI)},
  year={2025}}
```
-->

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"VessQC" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin
[file an issue]: https://github.com/MMV-Lab/VessQC/issues
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/