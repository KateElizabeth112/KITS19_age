import json
import os
import numpy as np
import pickle as pkl

local = False
if local:
    root_dir = "/Users/katecevora/Documents/PhD/data/KITS19/"
else:
    root_dir = "/rds/general/user/kc2322/projects/cevora_phd/live/kits19"

input_data_dir = os.path.join(root_dir, "data")
output_data_dir = os.path.join(root_dir, "FullDataset", "imagesTr")
output_label_dir = os.path.join(root_dir, "FullDataset", "labelsTr")


def reformatMetadata():
    # Reformat metadata from json to pickle
    f = open(os.path.join(input_data_dir, 'kits.json'))

    # returns JSON object as
    # a dictionary
    data = json.load(f)

    case_id = []
    age = []
    gender = []
    bmi = []

    cases = os.listdir(output_data_dir)

    for sub in data:
        id = sub["case_id"][6:]
        if "case_" + id + "_0000.nii.gz" in cases:
            case_id.append(id)
            age.append(sub["age_at_nephrectomy"])
            gender.append(sub["gender"])
            bmi.append(sub["body_mass_index"])

    case_id = np.array(case_id)
    age = np.array(age)
    gender = np.array(gender)
    bmi = np.array(bmi)

    # convert gender to binary indicator
    gender_bin = np.zeros(gender.shape)
    gender_bin[gender == "female"] = 1

    print("Number of females: {}".format(np.sum(gender_bin)))
    print("Number of males: {}".format(gender_bin.shape[0] - np.sum(gender_bin)))

    metadata = {"id": case_id,
                "age": age,
                "gender": gender_bin,
                "bmi": bmi}

    # save
    f = open(os.path.join(root_dir, "metadata.pkl"), "wb")
    pkl.dump(metadata, f)
    f.close()

    print("Done")


def main():
    reformatMetadata()



if __name__ == "__main__":
    main()