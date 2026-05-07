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

For `sample_experiment.json`, the resulting chunk size will be `33.36 x 20 x 100 um`, which has a voxelwise size of `834 x 500 x 2500 vx` 

## 1. Mask

Vessels **cast shadows** on the cells below them. For a cone above each pixel, remove an amount of power (fluorescence) based on the number of voxels with a vessel and the relative power density for each layer above it.

Vessels have a complex influence on the emission
1. Vessels attenuate the signal, they "cast shadows"
   1. NAOMi calculates a "mask" that is used to modify the fluorescence before convolving with the PSF. This is performed by essentially convolving a light cone with the vessel occupancy map above the neuron. 
   2. It's not realistic for us to perform this convolution for image. First, the amount of memory we would have to load to have the entire vessel network in RAM would be intractable. Second, the convolution takes too long and prevents us from scaling
   3. Instead, we'll use some masks that NAOMi has created previously. We have 2D 500x500 masks that we can multiply by the fluorescence to 
2. Vessels shift the propagating wavefront, because light travels through it at a different speed than the tissue

## 2. Convolve

