import os
import signal
import json
from time import sleep
from pathlib import Path
from argparse import ArgumentParser
from multiprocessing import Pool

import numpy as np
from numpy.lib.format import open_memmap

from caveclient import CAVEclient
from cloudvolume import CloudVolume, Bbox, Vec


def download_region(seg_cv, dataset_dir: Path, idx, region_bbox: Bbox, overwrite, **kwargs):

    region_root = dataset_dir / f'chunk_{idx[0]}_{idx[1]}_{idx[2]}'
    print(f'Downloading region # { idx[-1] } to: { region_root }')
    region_root.mkdir(parents=True, exist_ok=True)

    filenames = list(map(lambda cell_type: region_root / f'{cell_type}_{idx[3]}.npy', kwargs.keys()))
    exists = True
    for fname in filenames:
        exists &= fname.exists()

    if exists and (not overwrite):
        return True
    
    region_isovx_sz = (region_bbox / Vec(40,40,40)).size().astype(int)

    block = None
    for _ in range(3):
        try:
            block = seg_cv.image.download(region_bbox, mip=0)
            break
        except Exception as e:
            print(f'Error downloading block from CloudVolume: {e!r}. Retrying soon...')
            sleep(15)

    if block is None:
        print(f'Could not download block after 3 attempts. Please try again later.')
        return False

    print(f'Block size: { block.nbytes / (1024 ** 3) } GB')
    assert block.ndim == 4

    for cell_type, cell_ids in kwargs.items():
        filename = region_root / f'{cell_type}_{idx[3]}.npy'
        print(f'Creating region `{ filename }`')

        ids = np.asarray(cell_ids, dtype=block.dtype)

        cell_mask = np.isin(block, ids)
        cell_mask = np.any(cell_mask, axis=-1)
        assert cell_mask.ndim == 3
        sx, sy, sz = cell_mask.shape

        assert (not sx % 5) and (not sy % 5), f"Region shape must be a multiple of 5! { cell_mask.shape }"

        region = cell_mask.reshape(sx // 5, 5, sy // 5, 5, sz).sum(axis=(1, 3), dtype=np.uint8)
        assert np.all(region.shape == region_isovx_sz)

        if not np.any(region):
            print('No cells found in this region!')

        np.save(filename, region)

    return True


def assemble_regions(seg_cv, dataset_dir: Path, idx, chunk_size, cell_type, overwrite=False):
    filename = dataset_dir / f'chunk_{idx[0]}_{idx[1]}_{idx[2]}' / f'{cell_type}.npy'
    print(f'Assembling regions for chunk: { filename }')

    if filename.exists() and not overwrite:
        return True
    
    region_paths = [dataset_dir / f'chunk_{idx[0]}_{idx[1]}_{idx[2]}' / f'{cell_type}_{i}.npy' for i in range(25)]
    
    shapes = []
    for path in region_paths:
        region = np.load(path, mmap_mode='r')
        shapes.append(region.shape)

    x_sizes = { s[0] for s in shapes }
    y_sizes = { s[1] for s in shapes }
    assert len(x_sizes) == 1
    assert len(y_sizes) == 1

    outshape = (shapes[0][0], shapes[0][1], sum(s[2] for s in shapes))

    size_isovx = (chunk_size / Vec(40, 40, 40)).astype(int)
    size_isovx.unit = 'isovx'

    assert np.all(outshape == size_isovx)    
    
    chunk = open_memmap(filename, mode='w+', dtype=np.uint8, shape=outshape)
    print(f'Full chunk size: { chunk.nbytes / (1024 ** 3) } GB')
    
    z0 = 0
    for path in region_paths:
        region = np.load(path, mmap_mode='r')
        # assert (np.min(region) >= 0) and (np.max(region) <= 25)
        z1 = z0 + region.shape[-1]
        chunk[:,:,z0:z1] = region
        z0 = z1
        del region

    chunk.flush()

    del chunk
    return True


def clean_regions():
    pass


def download_chunk(seg_cv, dataset_dir: Path, idx: tuple, volume_bbox: Bbox, chunk_size: Vec, neurons, vessels, overwrite=False):
    seg_cv = CloudVolume('precomputed://https://storage.googleapis.com/iarpa_microns/minnie/minnie65/seg_m1300', progress=False, use_https=True)

    neurons_name = dataset_dir / f'chunk_{idx[0]}_{idx[1]}_{idx[2]}' / 'neurons.npy'
    vessels_name = dataset_dir / f'chunk_{idx[0]}_{idx[1]}_{idx[2]}' / 'vessels.npy'
    print(f'Downloading chunk to: { neurons_name.parent }')
    
    if (neurons_name.exists() and vessels_name.exists()) and not overwrite:
        return True

    chunk_origin = volume_bbox.minpt + (idx * chunk_size)
    chunk_bbox = Bbox(
        chunk_origin,
        chunk_origin + chunk_size,
        unit='nm'
    )
    chunk_bbox = Bbox.clamp(chunk_bbox, volume_bbox)
    chunk_size = chunk_bbox.size()

    region_size = Vec(chunk_size[0], chunk_size[1], np.ceil(chunk_size[2] / 25))

    download_ok = True

    for i in range(25):
        region_bbox = Bbox(
            chunk_origin +                         ((0,0,i  )*region_size),
            chunk_origin + ((1,1,0)*region_size) + ((0,0,i+1)*region_size),
            unit='nm'
        )
        region_bbox = Bbox.clamp(region_bbox, volume_bbox)
        download_ok &= download_region(seg_cv, dataset_dir, (*idx, i), region_bbox, overwrite, neurons=neurons, vessels=vessels)
        
    if not download_ok:
        print(f'Some regions in this chunk could not be downloaded. Please try again -- { neurons_name.parent }')
        return False
    
    assemble_regions(seg_cv, dataset_dir, idx, chunk_size, 'neurons', overwrite)
    assemble_regions(seg_cv, dataset_dir, idx, chunk_size, 'vessels', overwrite)

    clean_regions()


def download_chunk_star(args):
    return download_chunk(*args)


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

# Don't force `chunk_size` to perfectly tesselate the `volume_bbox`. Instead, let the
# overhanging chunks have a smaller size
def download_volume(seg_cv, dataset_dir: Path, volume_bbox: Bbox, chunk_size: Vec, neurons: list, vessels: list, overwrite=False, n_jobs=1):
    n_chunks = np.ceil(volume_bbox.size() / chunk_size).astype(int)
    chunk_idxs = []

    for xx in range(np.ceil(n_chunks[0])):
        for yy in range(np.ceil(n_chunks[1])):
            for zz in range(np.ceil(n_chunks[2])):
                chunk_idxs.append((xx, yy, zz))

    # for idx in chunk_idxs:
    #     download_chunk(seg_cv, dataset_dir, idx, volume_bbox, chunk_size, neurons, vessels, overwrite)

    args = [
        (None, dataset_dir, idx, volume_bbox, chunk_size, neurons, vessels, overwrite)
        for idx in chunk_idxs
    ]

    p = Pool(n_jobs, initializer=init_worker)
    try:
        for ok in p.imap_unordered(download_chunk_star, args, chunksize=1):
            if not ok:
                print('A chunk failed!')
    except KeyboardInterrupt:
        print('Keyboard Interrupt received, terminating...')
        p.terminate()
        p.join()
        raise
    except Exception:
        p.terminate()
        p.join()
        raise
    else:
        p.close()
        p.join()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--file', default=Path("D:\\JHU\\BDD\\SpineDetect-Clean\\Data\\normal-volume-size\\sample_experiment.json"))
    parser.add_argument('--overwrite', action='store_true', default=False)
    parser.add_argument('--n', type=int, default=1)

    args = parser.parse_args()

    experiment = None
    exp_filepath = Path(args.file)
    with open(exp_filepath, 'r') as exp_file:
        experiment = json.load(exp_file)

    neurons = None
    neuron_filepath = exp_filepath.parent / "neurons.txt"
    with open(neuron_filepath, 'r') as neuron_file:
        neurons = [
            np.uint64(line.strip())
            for line in neuron_file
            if line.strip()
        ]


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
            exit(1)
        
    experiment['chunk_size'] = chunk_size.tolist()
    with open(exp_filepath, 'w') as exp_file:
        json.dump(experiment, exp_file, indent=4)

    
    # seg_cv = CloudVolume('precomputed://https://storage.googleapis.com/iarpa_microns/minnie/minnie65/seg_m1300', progress=False, use_https=True)
    seg_cv = None
    download_volume(seg_cv, exp_filepath.parent, bbox, chunk_size, neurons, experiment['microns']['vessels'], args.overwrite, args.n)
