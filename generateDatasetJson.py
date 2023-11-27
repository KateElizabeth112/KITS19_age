# Generate the dataset.json file
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
import argparse
import os

# argparse
parser = argparse.ArgumentParser(description="Just an example",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-r", "--root_dir", default='/vol/biomedic3/kc2322/data/AMOS_3D', help="Root directory for nnUNet")
parser.add_argument("-n", "--dataset_name", default='Dataset703_Set3', help="Name of the dataset")
parser.add_argument("-tc", "--training_cases", default=120)
args = vars(parser.parse_args())

# set up variables
ROOT_DIR = args['root_dir']
DS_NAME = args['dataset_name']
TC = args['training_cases']

output_dir = os.path.join(ROOT_DIR, "nnUNet_raw/{}".format(DS_NAME))
imagesTr_dir = os.path.join(ROOT_DIR, "nnUNet_raw/{}/imagesTr".format(DS_NAME))
imagesTs_dir = os.path.join(ROOT_DIR, "nnUNet_raw/{}/imagesTs".format(DS_NAME))

channel_names = {0: "CT"}

labels = {"background": 0,
          "kidney": 1,
          "tumor": 2}

file_ending = ".nii.gz"

generate_dataset_json(str(output_dir),
                      channel_names,
                      labels,
                      int(TC),
                      file_ending)


