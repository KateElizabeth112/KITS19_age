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

    case_id_all = []
    age_all = []
    gender_all = []
    bmi_all = []

    cases = os.listdir(output_data_dir)

    for sub in data:
        id = sub["case_id"][6:]
        if "case_" + id + "_0000.nii.gz" in cases:
            case_id_all.append(id)
            age_all.append(sub["age_at_nephrectomy"])
            gender_all.append(sub["gender"])
            bmi_all.append(sub["body_mass_index"])

    case_id_all = np.array(case_id_all)
    age_all = np.array(age_all)
    gender_all = np.array(gender_all)
    bmi_all = np.array(bmi_all)

    # convert gender to binary indicator
    gender_bin_all = np.zeros(gender_all.shape)
    gender_bin_all[gender_all == "female"] = 1

    print("Number of females: {}".format(np.sum(gender_bin_all)))
    print("Number of males: {}".format(gender_bin_all.shape[0] - np.sum(gender_bin_all)))

    # Now only save the examples in the input folder
    case_id = []
    age = []
    sex = []
    bmi = []

    filenames = os.listdir(output_data_dir)

    for fn in filenames:
        if fn.endswith(".nii.gz"):
            id = fn[5:9]

            if not (id in case_id_all):
                print("subject {} not found in metadata".format(id))

            case_id.append(id)
            age.append(age_all[case_id_all == id][0])
            sex.append(gender_bin_all[case_id_all == id][0])
            bmi.append(bmi_all[case_id_all == id][0])

    metadata = {"id": case_id,
                "age": age,
                "sex": sex,
                "bmi": bmi}

    # save
    f = open(os.path.join(root_dir, "info.pkl"), "wb")
    pkl.dump(metadata, f)
    f.close()

    print("Done")


def main():
    reformatMetadata()



if __name__ == "__main__":
    main()