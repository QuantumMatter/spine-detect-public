import json
from pathlib import Path
from argparse import ArgumentParser

import numpy as np

from caveclient import CAVEclient
from cloudvolume import CloudVolume, Bbox, Vec

from sd_utils.microns import get_all_cts

# Make sure that bbox has the appropriate units before calling
def generate_neuron_list(client, bbox, cell_types, p):
    seg_cv = CloudVolume('precomputed://https://storage.googleapis.com/iarpa_microns/minnie/minnie65/seg_m1300', progress=False, use_https=True)

    all_cts = get_all_cts(client)
    matching_cts = all_cts[all_cts["cell_type"].isin(cell_types)].copy()
    matching_cell_ids = set(matching_cts["pt_root_id"].astype(np.uint64).map(int))

    cells_in_bbox = {int(cid) for cid in seg_cv.unique(bbox, mip=3)}
    cells_in_bbox.discard(0)

    matching_cells_in_bbox = matching_cell_ids & cells_in_bbox

    n_neurons = np.random.binomial(len(matching_cells_in_bbox), p)
    neurons = np.random.choice(list(matching_cells_in_bbox), n_neurons, replace=False)

    return neurons


# Assumes that the `file` is in the dataset root 
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--file', default=Path("D:\\JHU\\BDD\\SpineDetect-Clean\\Data\\normal-volume-size\\sample_experiment.json"))
    parser.add_argument('--overwrite', action='store_true', default=False)

    args = parser.parse_args()

    experiment = None
    exp_filepath = Path(args.file)
    with open(exp_filepath, 'r') as exp_file:
        experiment = json.load(exp_file)

    neuron_filepath = exp_filepath.parent / "neurons.txt"
    if neuron_filepath.exists() and not args.overwrite:
        print("Not overwiting neurons.txt that already exists!")
        exit(1)

    client = CAVEclient(experiment['microns']['dataset'])
    client.version = experiment['microns']['materialization']

    bbox = Bbox(
        Vec(*experiment['microns']['volume'][0]),
        Vec(*experiment['microns']['volume'][1]),
        unit='nm'
    )
    cell_types = experiment['microns']['cell_types']
    p = experiment['microns']['neuron_density']

    neurons = generate_neuron_list(client, bbox, cell_types, p)

    with open(neuron_filepath, 'w') as neuron_file:
        neuron_file.write("\n".join([str(n) for n in neurons]))

    