"""
Script to download and split all datasets
"""

import pdb
import pickle
import random
import os
import argparse
import json
import nibabel
import numpy as np
import pandas as pd
import copy
import time
from urllib.request import urlretrieve
import subprocess


def download_elnino(args):
    files = {
        "tao-all2.dat.gz": "https://archive.ics.uci.edu/ml/machine-learning-databases/el_nino-mld/tao-all2.dat.gz",
        "tao-all2.missing.gz": "https://archive.ics.uci.edu/ml/machine-learning-databases/el_nino-mld/tao-all2.missing.gz",
        "elnino.gz": "https://archive.ics.uci.edu/ml/machine-learning-databases/el_nino-mld/elnino.gz",
        "tao-all2.col": "https://archive.ics.uci.edu/ml/machine-learning-databases/el_nino-mld/tao-all2.col",
        "elnino.col": "https://archive.ics.uci.edu/ml/machine-learning-databases/el_nino-mld/elnino.col",
    }
    os.makedirs(os.path.join(args.output, "elnino"), exist_ok=True)
    for name, url in files.items():
        location = os.path.join(args.output, "elnino", name)
        location, _ = urlretrieve(url, location)
        if location.endswith(".gz"):
            unzipped_location = location[:-3]

            cmd = ["gzip", "-kd", location]
            print("Using subprocess to extract zipfile.")
            print("$" + " ".join(cmd))
            subprocess.call(cmd)
    return


RECOGNIZED_DATASETS = {"elnino": download_elnino}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs='+', type=str, choices=["all"] + list(RECOGNIZED_DATASETS.keys()), help="Which data")
    parser.add_argument("-o", "--output", type=str, default="data/", help="Directory within which to save datasets")
    args = parser.parse_args()
    for name in args.names:
        RECOGNIZED_DATASETS[name](args)
