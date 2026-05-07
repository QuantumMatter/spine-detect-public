# Spine Detect

The purpose of this project is to train a deep learning model to automatically segement dendritic spines in 2-Photon microscopy images. To do this, we generate a large synthetic traing dataset from MICrONS that is 3 orders of magnitude larger than previously published methods.

## Project Setup

Note that the environment uses the `tensorflow[and-cuda]` Python package, which is only supported on Linux. If you're using Windows, you must use WSL2. If you're on Mac, update the requirement in `environment.yml` with the approprate package name (probably just `tensorflow`)

```bash
# Setup the virtual environment
conda env create -f environment.yml

# Activate the environment
conda activate spine-detect

# Install the local packages
pip install -e ./Shared/sd-utils

# Install HKS package
pip install git+https://github.com/bdpedigo/meshmash@e6b9915
```

Once your environment is established, make sure you're authenticated with the MICrONS platform by following the [instructions on their website](https://tutorial.microns-explorer.org/quickstart_notebooks/01-caveclient-setup.html).


## Project Structure
```
├── 1-Bootstrap
│   ├── 1-HKS-Classifier
|   |   |── README
|   |   |── src
|   |   └── run
|
├── 2-Dataset-Generation
│   ├── 1-Setup
│   |── 2-Label
|   |   |── 1-Download
|   |   |── 2-HKS
|   |   |── 3-Classification
|   |   ├── 4-Mask
|   |   └── 5-Merge
│   └── 3-Feature
|
├── 3-Training
|
|── Shared
│   └── sd-utils
|
└── Data
    |── Bootstrap
    |   ├── HKS
    |   └── NAOMi
    |
    |── Label
    |   └── <pt_root_id>
    |       ├── mesh_high_res
    |       ├── mesh_low_res
    |       ├── em.npy
    |       ├── hks_features.csv
    |       └── mask.tiff
    |
    └── <Dataset-ID>                    # For every invocation of `Dataset-Generation/Feature`
        |── neurons.txt
        |── experiment.json
        └── <Chunk-ID>                 # For every "imaging volume" (chunks that span the volume)
            |── merged_mask.tiff
            |── vessels.npy
            |── em.npy
            └── <Image-ID>              # For every microscope setting
                |── naomi.json          # Laser power, lens NA, stride (resolution), etc
                |── conv.tiff           # Output from convolving with PSF
                |── feature.tiff        # Final synthetic image (conv.tiff + noise)
                └── label.tiff          # Final ground truth segmentation (merged_mask.tiff + stride)


```