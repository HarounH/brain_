import pickle
import numpy as np
import pandas as pd
from nilearn.image import load_img, resample_to_img, crop_img, threshold_img, math_img
import nilearn.datasets
import nibabel
import torch
import time


original_brain_mask = nibabel.load("/data/hcp/hcp_mask.nii.gz")

original_dataframe_csv_file = "/data/neurovault/dataframe.csv"

if __name__ == '__main__':
    tic = time.time()
    df = pd.read_csv(original_dataframe_csv_file)
    brk = False
    n = df.shape[0]
    new_contrasts = []
    new_tasks = []
    for i in range(n):
        if df.iloc[i].study in ["camcan", "brainomics"]:
            fn = df.iloc[i].z_map
            img = nibabel.load(fn)
            img = math_img("img1 * img2", img1=resample_to_img(img, original_brain_mask), img2=original_brain_mask)
            nibabel.save(img, fn)
            if brk:
                import pdb; pdb.set_trace()
        if not(df.iloc[i].contrast.startswith(df.iloc[i].study)):
            new_contrasts.append(df.iloc[i].study + "-" + df.iloc[i].contrast)
        else:
            new_contrasts.append(df.iloc[i].contrast)

        if not(df.iloc[i].task.startswith(df.iloc[i].study)):
            new_tasks.append(df.iloc[i].study + "-" + df.iloc[i].task)
        else:
            new_tasks.append(df.iloc[i].task)
        # print("[{}/{}]".format(i, n))
    df.contrast = new_contrasts
    df.task = new_tasks
    import pdb; pdb.set_trace()
    df.to_csv(original_dataframe_csv_file)
    print("Completed in {}s".format(time.time() - tic))
