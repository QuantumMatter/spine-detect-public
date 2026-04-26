import json
from time import sleep
from pathlib import Path
from argparse import ArgumentParser

import numpy as np

from caveclient import CAVEclient
from cloudvolume import CloudVolume, Bbox, Vec


def download_region(client, dataset_dir: Path, idx, region_bbox: Bbox, overwrite, **kwargs):
    assert len(kwargs) == 1
    cell_type = list(kwargs.keys())[0]
    cell_ids = kwargs.get(cell_type)

    region_name = dataset_dir / f'chunk_{idx[0]}_{idx[1]}_{idx[2]}' / f'{cell_type}_{idx[3]}.npy'
    print(f'Downloading region to: { region_name }')
    if not region_name.parent.exists():
        region_name.parent.mkdir()

    if region_name.exists() and not overwrite:
        return True
    
    region_isovx_sz = (region_bbox / Vec(40,40,40)).size().astype(int)
    region = np.zeros(region_isovx_sz)

    retries = 0
    while retries < 3:
        retries += 1
        try:
            seg_cv = CloudVolume('precomputed://https://storage.googleapis.com/iarpa_microns/minnie/minnie65/seg_m1300', progress=False, use_https=True)
            block = seg_cv.image.download(region_bbox, mip=0)
            break
        except:
            print('Error downloading block from CloudVolume. Retrying soon...')
            sleep(15)

    if block is None:
        print(f'Could not download block after { retries } attempts. Please try again later.')
        return False

    print(f'Block size: { block.nbytes / (1024 ** 3) } GB')

    for cell_id in cell_ids:
        print(f'Processing cell id: { cell_id }')
        
        cell_mask = (block == np.uint64(cell_id)).astype(np.int8)
        if not np.any(cell_mask): continue

        pooled = np.zeros_like(region)
        for xx in range(region.shape[0]):
            for yy in range(region.shape[1]):
                window = cell_mask[
                    xx*5:(xx+1)*5,
                    yy*5:(yy+1)*5,
                    :
                ]
                pooled[xx,yy,:] = np.sum(window, axis=(0,1)).squeeze()

        region += pooled

    np.save(region_name, region)
    return True


def assemble_regions(client, dataset_dir: Path, idx, chunk_size, cell_type, overwrite=False):
    filename = dataset_dir / f'chunk_{idx[0]}_{idx[1]}_{idx[2]}' / f'{cell_type}.npy'
    print(f'Assembling regions for chunk: { filename }')

    if filename.exists() and not overwrite:
        return True
    
    size_isovx = (chunk_size / Vec(40, 40, 40)).astype(int)
    size_isovx.unit = 'isovx'

    # chunk_depth = 0
    # for i in range(25):
    #     region_name = dataset_dir / f'chunk_{idx[0]}_{idx[1]}_{idx[2]}' / f'{cell_type}_{i}.npy'
    #     region = np.load(region_name)
    #     chunk_depth += region.shape[-1]
    #     del region
    # assert chunk_depth == size_isovx[3]
    # print(f'Chunk shape: { chunk.shape } (isovx)')
    
    chunk = np.zeros(size_isovx, dtype=np.int8)
    print(f'Full chunk size: { chunk.nbytes / (1024 ** 3) } GB')
    
    depth = 0
    for i in range(25):
        region_name = dataset_dir / f'chunk_{idx[0]}_{idx[1]}_{idx[2]}' / f'{cell_type}_{i}.npy'
        region = np.load(region_name).astype(np.single)
        # assert (np.min(region) >= 0) and (np.max(region) <= 25)
        chunk[:,:,depth:depth+region.shape[-1]] = region
        del region

    np.save(filename, chunk)

    del chunk
    return True


def clean_regions():
    pass


def download_chunk(client, dataset_dir: Path, idx: tuple, volume_bbox: Bbox, chunk_size: Vec, neurons, vessels, overwrite=False):
    neurons_name = dataset_dir / f'chunk_{idx[0]}_{idx[1]}_{idx[2]}' / 'neurons.npy'
    vessels_name = dataset_dir / f'chunk_{idx[0]}_{idx[1]}_{idx[2]}' / 'vessels.npy'
    print(f'Downloading chunk to: { neurons_name.parent }')
    
    if (neurons_name.exists() and vessels_name.exists()) and not overwrite:
        return True


    chunk_origin = volume_bbox.minpt + (idx * chunk_size)
    region_size = Vec(chunk_size[0], chunk_size[1], np.ceil(chunk_size[2] / 25))

    download_ok = True

    for i in range(25):
        region_bbox = Bbox(
            chunk_origin +                         ((0,0,i  )*region_size),
            chunk_origin + ((1,1,0)*region_size) + ((0,0,i+1)*region_size),
            unit='nm'
        )
        region_bbox = Bbox.clamp(region_bbox, volume_bbox)
        download_ok &= download_region(client, dataset_dir, (*idx, i), region_bbox, overwrite, neurons=neurons)
        download_ok &= download_region(client, dataset_dir, (*idx, i), region_bbox, overwrite, vessels=vessels)

    if not download_ok:
        print(f'Some regions in this chunk could not be downloaded. Please try again -- { neurons_name.parent }')
        return False
    
    assemble_regions(client, dataset_dir, idx, chunk_size, 'neurons', overwrite)
    assemble_regions(client, dataset_dir, idx, chunk_size, 'vessels', overwrite)

    clean_regions()


# Don't force `chunk_size` to perfectly tesselate the `volume_bbox`. Instead, let the
# overhanging chunks have a smaller size
def download_volume(client, dataset_dir: Path, volume_bbox: Bbox, chunk_size: Vec, neurons: list, vessels: list, overwrite=False):
    n_chunks = np.ceil(volume_bbox.size() / chunk_size).astype(int)
    chunk_idxs = []

    for xx in range(np.ceil(n_chunks[0])):
        for yy in range(np.ceil(n_chunks[1])):
            for zz in range(np.ceil(n_chunks[2])):
                chunk_idxs.append((xx, yy, zz))

    for idx in chunk_idxs:
        download_chunk(client, dataset_dir, idx, volume_bbox, chunk_size, neurons, vessels, overwrite)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--file', default=Path("D:\\JHU\\BDD\\SpineDetect-Clean\\Data\\normal-volume-size\\sample_experiment.json"))
    parser.add_argument('--overwrite', action='store_true', default=False)

    args = parser.parse_args()

    experiment = None
    exp_filepath = Path(args.file)
    with open(exp_filepath, 'r') as exp_file:
        experiment = json.load(exp_file)

    neurons = None
    neuron_filepath = exp_filepath.parent / "neurons.txt"
    with open(neuron_filepath, 'r') as neuron_file:
        neurons = neuron_file.readlines()
        neurons = list(map(np.int64, neurons))


    client = CAVEclient(experiment['microns']['dataset'])
    client.version = experiment['microns']['materialization']

    bbox = Bbox(
        Vec(*experiment['microns']['volume'][0]),
        Vec(*experiment['microns']['volume'][1]),
        unit='nm'
    )

    chunk_bytes = experiment['chunk_gb'] * (1024 ** 3)
    chunk_nelem = chunk_bytes / 4 # Single precision

    bbox_isovx = bbox / Vec(40, 40, 40)
    bbox_isovx.unit = 'isovx'

    print(bbox.size())
    print(bbox_isovx.size())

    n_chunks = bbox_isovx.volume() / chunk_nelem
    if n_chunks < 1:
        n_chunks = 1
        chunk_size = bbox.size() 
    else:
        # Try to split along just the fine resolution dimensions
        xy_split = np.ceil(np.sqrt(n_chunks))
        isobbox_size = bbox_isovx.size()
        iso_chunk_size = Vec(np.ceil(isobbox_size[0] / xy_split), np.ceil(isobbox_size[1] / xy_split), isobbox_size[2])
        chunk_size = iso_chunk_size * Vec(40, 40, 40)
        chunk_size = chunk_size.astype(int)
        chunk_size.unit = 'nm'

    print(chunk_size)
    print(f'Estimated Chunk Size: { ((chunk_size / Vec(40, 40, 40)).rectVolume() * 4) / (1024 ** 3) } GB')

    if 'chunk_size' in experiment:
        if (np.any(chunk_size != experiment['chunk_size'])) and not args.overwrite:
            print('Error! Not overriding `chunk_size` set in a previous run. Please run again with the --overwrite flag')
        
    experiment['chunk_size'] = chunk_size.tolist()
    with open(exp_filepath, 'w') as exp_file:
        json.dump(experiment, exp_file, indent=4)

    download_volume(client, exp_filepath.parent, bbox, chunk_size, neurons, experiment['microns']['vessels'], args.overwrite)