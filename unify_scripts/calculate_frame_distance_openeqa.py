import os
root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),  "..")
import sys
sys.path.append(root_dir)
import argparse
import json
import pickle as pkl
import statistics

def percentage2frame(duration, x):
    x = int(x / 100 * duration )
    return x

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_file",
        type=str,
        help="json_file",
    )

    args = parser.parse_args()

    return args

def distance(gt, pred):
    distance = []
    for i in range(len(pred)):
        dist = abs(pred[i] - gt[i])
        distance.append(dist)
    return statistics.mean(distance)

def convert(duration, x):
    x = x / duration * 100
    x = min(round(x), 99)
    return x

if __name__ == "__main__":

    args = parse_args()

    json_file = args.json_file
    phase = json_file.split('-')[0]
    predicted_frame_len = int(json_file.split('-')[1])
    q_or_a = json_file.split('_')[-1][:-5]

    if phase == "train":
        dset = "openeqa_train"
    elif phase == "val":
        dset = "openeqa_val"
    else:
        print("not supported phase ", phase)

    if predicted_frame_len == 2:
        exp_names = ['train_2e-5_ckpt-500_2-frame_', 'train_ckpt-400_2-frame_']
    elif predicted_frame_len == 6:
        exp_names = ['train_ckpt-220_6-frame_']
    exp_names = [name+q_or_a for name in exp_names]

    if phase == "val":
        exp_names = [name.replace("train", "val") for name in exp_names]

    for exp_name in exp_names:
        print(exp_name)

        exp_dir = os.path.join("/home/fengchan/stor/dataset/sparse_bwd/predicted_best_frames/", dset, exp_name)
        gt_dir = os.path.join("/mnt/opr/fengchan/dataset/sparse_bwd/psuedo_selected_frames/open_eqa_2/")

        total_distance = 0
        js = json.load(open(f"./data/open_eqa/{args.json_file}"))
        num = 0
        for data in js:
            num += 1
            id = data["id"]
            qid = data["qid"]
            duration = data["meta"]["duration"]
            qid = str(qid)
            groundtruths = data["meta"]["token"].values()
            groundtruths = list(groundtruths)
            groundtruths = list(map(float, groundtruths))
            groundtruths = [convert(duration, gt) for gt in groundtruths]

            prediction = pkl.load(open(os.path.join(exp_dir, f"{id}.pkl"), "rb"))[qid]
            prediction = prediction['frame_indices'][:predicted_frame_len]
            prediction = [convert(duration, pred) for pred in prediction]
            print(groundtruths, prediction)
            try:
                # prediction = list(map(int, prediction))
                total_distance += distance(groundtruths, prediction)
            except:
                continue
        print(total_distance / num)

        



