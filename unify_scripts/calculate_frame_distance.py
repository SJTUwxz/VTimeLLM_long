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
    parser.add_argument("--dset", type=str, required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument(
        "--predict_segment",
        action="store_true",
        help="whether to use selected chunks",
    )
    parser.add_argument(
        "--predict_frames",
        action="store_true",
        help="whether to use selected chunks",
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
    x = str(min(round(x), 99))
    if len(x) == 1:
        x = "0" + x
    return x

if __name__ == "__main__":

    args = parse_args()

    dset = args.dset
    exp_name = args.exp_name

    exp_dir = os.path.join("/home/fengchan/stor/dataset/sparse_bwd/vtimellm_predictions/", dset, exp_name)

    total_distance = 0
    num = 0
    js = json.load(open(f"./data/{dset}.json"))
    for data in js:
        num = num + 1
        id = data["id"]
        qid = data["qid"]
        duration = data["meta"]["duration"]
        qid = str(qid)
        if id not in qid:
            qid = id + qid
        groundtruths = data["meta"]["token"].values()
        groundtruths = list(groundtruths)
        groundtruths = list(map(float, groundtruths))
        groundtruths = [convert(duration, gt) for gt in groundtruths]
        predicted_frame_len = len(groundtruths)
        
        prediction = pkl.load(open(os.path.join(exp_dir, f"{id}.pkl"), "rb"))[qid]
        prediction = prediction['prediction'].split(' ')[:predicted_frame_len]
        try:
            prediction = list(map(int, prediction))
            total_distance += distance(groundtruths, prediction)
        except:
            continue
        print(groundtruths, prediction)
    print(total_distance / num)

        



