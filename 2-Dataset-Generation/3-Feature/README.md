# Dataset / Feature

## 1. Download

* Downloads segmentation data from MICrONS for the selected neurons
* Downsamples to an isotropic 40nm voxel size
* Rearranges dimensions from (x=lateral, y=depth, z=lateral) to (y=depth, x=lateral, z=lateral)
* Write smaller chunks to disk to easier loading

1. A target chunk size is calculated based on an amount of memory (ie 4 GB)
2. Update `experiment.json` with the chunk size for later steps
3. For each chunk
    1. A small region is calculated as 1/25th of the chunk size
    2. For each region
        1. For each cell id
            1. Download the segmentation from MICrONS
            2. Filter for voxels that match the cell id
            3. Downsample 5x the first two dimensions by sum-pooling
            4. Accumulate the values in a regional matrix
        2. Save the region to disk
    3. Concatenate all of the regions together
    4. Save the chunk to the dataset folder
    5. Clean up region files