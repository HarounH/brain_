# brain_

Everything in this repo is run from the project root as a module.
python -m <module>


# Getting Data
There are a couple of steps.
## Getting data:
First, we need to get resting state data seperately.
1. `python -m data.download_hcp [absolute path]` will download and resample HCP1200 data.
2. `python -m data.get_data all --resample` will fetch all neurovault data, resample it, create dataframe pointing to all files.
3. [gcn] `python m data.parcellations [n=30000] --rfmri` will run a parcellation on HCP downsampled data and save it at the file paths specified in data.constants.py


# Convolutions
Directory structure is self explanatory.
SHACGAN is the primary model. It creates a generator, discriminator and gradient penalty of specified version. It also supports DataParallel through the .toggle_dataparallel() line.
However, the SHACGAN itself doesn't have a forward function since that wouldn't make too much sense.
```
python -m conv.main [datasetnames] --cuda --dp -ve 100 -se 1000
```

# Graph
