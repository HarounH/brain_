import os
import sys
import numpy as np
import nilearn
import nilearn.image
import nibabel
import numpy as np
from nilearn.image import load_img, resample_to_img, resample_img, crop_img, threshold_img, math_img
from nilearn.datasets import load_mni152_template, load_mni152_brain_mask
import boto3
import botocore
import pdb
import time
from multiprocessing import Pool
from data.constants import hcp_subject_list, brain_mask

# assert(os.path.isfile("~/.aws/credentials"))

s3 = boto3.resource('s3')
# Stuff to download from
BUCKET_NAME = 'hcp-openaccess'  # replace with your bucket name
base_dir = "HCP_1200/"
subject_ids = hcp_subject_list
sub_files = {k: "MNINonLinear/Results/rfMRI_REST{}/rfMRI_REST{}.nii.gz".format(k, k) for k in ["1_LR"]}  # ["1_LR", "2_LR", "1_RL", "2_RL"]}

target_location = sys.argv[1]
if target_location[-1] != "/":
    target_location = target_location + "/"
os.makedirs(target_location, exist_ok=True)
os.makedirs(os.path.join(target_location, "original"), exist_ok=True)
os.makedirs(os.path.join(target_location, "downsampled"), exist_ok=True)


def file_namer(s="", t="", k=""):
    return "rfMRI_REST_{}_{}{}.nii.gz".format(s, k, "_" + str(t) if t!="" else "")


def fetch_subject(param_pair):
    i, subject = param_pair
    bucket = s3.Bucket(BUCKET_NAME)  # No idea what this is.
    print("Starting {}:{}".format(i, subject))
    for k, file_ in sub_files.items():
        start = time.time()
        KEY = base_dir + str(subject) + "/" + file_
        download_path = os.path.join(target_location, "original", file_namer(s=subject, k=k))
        try:
            bucket.download_file(KEY, download_path)
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                print("The object does not exist.")
                continue
            else:
                print("stopped after {}:{}".format(i-1, hcp_subject_list[i-1]))
                continue
        print("Download[{}, {}] took {}s".format(i, k, time.time() - start))

        start = time.time()
        nimg = nibabel.load(download_path)
        data = nimg.get_data()
        for t in range(0, nimg.shape[-1], 10):
            img = math_img(
                "img1 * img2",
                img1=resample_to_img(
                    nibabel.Nifti1Image(
                        data[:, :, :, t], nimg.affine
                    ),
                    brain_mask
                ),
                img2=brain_mask,
            )
            downsampled_path = os.path.join(target_location, "downsampled", file_namer(s=subject, k=k, t=t))
            nibabel.save(img, downsampled_path)
        print("Downsampling [{}, {}] took {}s".format(i, k, time.time() - start))
        os.remove(download_path)
    print("Done with {}:{}".format(i, subject))

p = Pool(4)
subjects = hcp_subject_list  # [hcp_subject_list[0], hcp_subject_list[1]]
iss = range(len(subjects))
pool_params = zip(iss, subjects)
p.map(fetch_subject, pool_params)
