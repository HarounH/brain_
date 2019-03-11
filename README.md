# FGL
Everything in this repo is run from the project root as a module.
python -m <module>

## Requirements
Pytorch 1.0
scikit
nilearn
nibabel
torch-scatter (https://github.com/rusty1s/pytorch_scatter)
pycairo 

## Getting Data
### Neurovault datasets
`wget http://cogspaces.github.io/assets/data/hcp_mask.nii.gz`
`python -m data.get_data [dataset names] --fetch --stats` will save datasets to `/data/neurovault/`
`python -m data.upsample_data` will upsample camcan and brainomics to the MNI152 template downloaded above.
### HCP datasets
`python -m data.download_hcp [hcp location]`. Read the file to understand whats going on. (The variables you'll need to change to configure stuff is at the top of the code)

### Parcellation
`python -m data.parcellations number-of-parcels [directory containing hcp] [output destination]`

## Running stuff

### Classification Experiments
`sh run_all.sh`

### Autoencoder experiments
