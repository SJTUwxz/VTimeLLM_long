import os
root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),  "..")
import sys
sys.path.append(root_dir)
import argparse
import json
import pickle as pkl
import statistics
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_file",
        type=str,
        help="json_file",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="save dir",
        default='open_eqa_segments',

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
    js = json.load(open(f"./data/open_eqa/{args.json_file}"))
    save_path = os.path.join("/mnt/opr/fengchan/dataset/sparse_bwd/psuedo_selected_frames/", args.save_dir)
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, "hm3d-v0"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "scannet-v0"), exist_ok=True)
    num = 0
    video = {}
    for data in js:
        num += 1
        id = data["id"]
        qid = data["qid"]
        duration = data["meta"]["duration"]
        qid = str(qid)
        groundtruths = data["meta"]["token"].values()
        groundtruths = list(groundtruths)

        if id not in video:
            video[id] = {}
        video[id][qid] = {}

        segments = []
        result = []
        for i, timestamp in enumerate(groundtruths):
            if i%2 == 0:
                start = timestamp
            if i%2 == 1:
                end = timestamp
                segments.append([start, end])


        num_frames_per_segment = int(6 / len(segments))
        for i, segment in enumerate(segments):
            try:
                start, end = segment

                if i == len(segments) - 1:
                    num_frames_per_segment = 6 - len(result)

                sampled_result = np.round(np.linspace(start, end, num_frames_per_segment)).astype(int)
                result.extend(sampled_result)
            except:
                print("error ", segment)

        assert len(result) == 6, "result length not enough"
        print(segments, result)
        
        video[id][qid]['frame_indices'] = result 

    for vid in video.keys():
        save_data = video[vid]
        pkl.dump(save_data, open(os.path.join(save_path, f"{vid}.pkl"), "wb"))

    print(save_path)



        



