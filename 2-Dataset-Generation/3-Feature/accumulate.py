import json
import re
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
from cloudvolume import Bbox, Vec
import tifffile as tf

regex = r"^\w+_(\d+)_(\d+)_(\d+)$"

def accumulate(dataset_dir: Path, volume_size, chunk_size, overwrite=False):
    acc_name = dataset_dir / 'volume.npy'
    tiff_name = dataset_dir / 'volume.tiff'
    if acc_name.exists() and not overwrite:
        return True

    v_iso_size = (volume_size / Vec(40, 40, 40)).astype(int)
    c_iso_size = (chunk_size / Vec(40, 40, 40)).astype(int)
    
    volume = np.zeros(v_iso_size, dtype=np.int8) - 1
    print(f'Full volume size: { volume.nbytes / (1024 ** 3) } GB')
    
    for chunk_name in dataset_dir.glob("chunk_*/neurons.npy"):
        m = re.match(regex, chunk_name.parent.name)
        if m is None:
            print(f'Path is not of the proper form! { chunk_name.parent.name }')
            continue
        chunk_coord = np.array(m.groups()).astype(int)

        print(f'Loading chunk ({ chunk_coord }): {chunk_name}')
        chunk = np.load(chunk_name)

        start = c_iso_size * chunk_coord
        end = start + chunk.shape

        x0, y0, z0 = start
        x1, y1, z1 = end

        volume[x0:x1, y0:y1, z0:z1] = chunk

        del chunk

    np.save(acc_name, volume)
    tf.imwrite(tiff_name, volume, compression='deflate')

    del volume


if __name__ == "__main__":
    default_file = Path(__file__).parent.parent.parent / 'Data' / 'normal-volume-size' / 'sample_experiment.json'

    parser = ArgumentParser()
    parser.add_argument('--file', default=default_file)
    parser.add_argument('--overwrite', action='store_true', default=False)

    args = parser.parse_args()

    experiment = None
    with open(args.file, 'r') as exp_file:
        experiment = json.load(exp_file)

    bbox = Bbox(
        Vec(*experiment['microns']['volume'][0]),
        Vec(*experiment['microns']['volume'][1]),
        unit='nm'
    )

    accumulate(Path(args.file).parent, bbox.size(), Vec(*experiment['chunk_size']))
