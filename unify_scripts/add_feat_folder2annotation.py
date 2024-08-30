import os
root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
import sys
sys.path.append(root_dir)
import json
from tqdm import tqdm

if __name__ == "__main__":

    stage2_json = "./data/vtimellm_train/stage2.json"
    momentor_json = "./data/vtimellm_train/momentor_6_questions.json"
    out_f = open("./data/vtimellm_train/merged_stage2_momentor.json", "w")

    output_list = []

    stage2 = json.load(open(stage2_json, "r"))
    momentor = json.load(open(momentor_json, "r"))

    output_list = []

    for item in stage2:
        item["feat100_folder"] = "./data/vtimellm_train/intern_clip_feat"
        item["feat400_folder"] = ""
        output_list.append(item)

    for item in momentor:
        item["feat100_folder"] = "/mnt/mir/datasets/vlm-datasets/momentor-10M/clip_features_100"
        item["feat400_folder"] = "/mnt/mir/datasets/vlm-datasets/momentor-10M/clip_features_400"
        output_list.append(item)


    json.dump(output_list, out_f, indent=6)
    out_f.close()
