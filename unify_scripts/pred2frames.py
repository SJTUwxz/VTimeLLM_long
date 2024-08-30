import os
root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),  "..")
import sys
sys.path.append(root_dir)
import argparse
import json
import pickle as pkl
import statistics
import random
from decord import VideoReader
import numpy as np

from glob import glob

def percentage2frame(duration, x):
    x = int(x / 100 * duration )
    return x

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predict_segment",
        action="store_true",
        help="whether to use selected chunks",
    )
    parser.add_argument(
        "--frames_num",
        type=int,
        help="whether to use selected chunks",
    )
    parser.add_argument(
        "--question_or_answer",
        type=str,
        choices=["question", "answer"]
    )
    parser.add_argument(
        "--subset",
        type=str,
        choices=["traintest", "val"]
    )
    parser.add_argument("--missing_prediction_complete", type=str, default='uniform', choices=["uniform", "reuse"])

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

def percentage2frame(duration, x):
    x = int(x / 100 * duration )
    return x

if __name__ == "__main__":

    args = parse_args()

    video_dir = "/mnt/mir/datasets/vlm-evaluation-datasets/nextqa/videos/"

    # dset should be nextqa_train or nextqa
    frames_num = args.frames_num
    question_or_answer = args.question_or_answer
    subset = args.subset
    missing_prediction_complete = args.missing_prediction_complete

    expected_frames_num = 6
    
    dset = "next_qa"
    if subset == "traintest":
        dset = dset + "_traintest"
    dataset = dset

    if args.predict_segment:
        dset = dset + f"_segment_{question_or_answer}"
        exp_dir = os.path.join("/home/fengchan/stor/dataset/sparse_bwd/vtimellm_predictions/", dset)
        save_dir = os.path.join("/home/fengchan/stor/dataset/sparse_bwd/predicted_best_frames/", dataset, f"{subset}_segment_{question_or_answer}")
        os.makedirs(save_dir, exist_ok=True)
    else:
        dset = dset + f"_{frames_num}-frame_{question_or_answer}"
        exp_dir = os.path.join("/home/fengchan/stor/dataset/sparse_bwd/vtimellm_predictions/", dset)
        save_dir = os.path.join("/home/fengchan/stor/dataset/sparse_bwd/predicted_best_frames/", dataset, f"{subset}_{frames_num}-frame_{question_or_answer}")
        os.makedirs(save_dir, exist_ok=True)

    print("predicted 6 best frames will be saved to ", save_dir)
    print("exp_dir ", exp_dir)

    for filename in glob(f'{exp_dir}/*/*.pkl', recursive=True):

        video_info = {}

        prediction = pkl.load(open(filename, "rb"))
        id = os.path.basename(filename)[:-4]
        for qid in prediction.keys():
            qid = str(qid)
            if id not in qid:
                qid = id + qid

            vr = VideoReader(os.path.join(video_dir, str(id)+'.mp4'))
            fps_vid = vr.get_avg_fps()
            duration = float(len(vr)) / fps_vid
            duration = int(duration)

            predicted_string = prediction[qid]['prediction']

            result = []
            # if predict frames
            if not args.predict_segment:
                preds = predicted_string.split(' ')[:frames_num]
                for frm in preds:
                    try:
                        frm = int(frm)
                        frm = percentage2frame(duration, frm)
                        result.append(frm)
                    except:
                        continue
                if missing_prediction_complete == "reuse":
                    if len(result) < expected_frames_num:
                        try:
                            sampled_result = random.sample(result, expected_frames_num-len(result))
                            result.extend(sampled_result)
                        except:
                            missing_prediction_complete = "uniform"

                if missing_prediction_complete == "uniform":
                    missing_frames_num = expected_frames_num - len(result)
                    if len(result) < expected_frames_num:
                        sampled_result = np.round(np.linspace(0, duration - 1, missing_frames_num)).astype(int)
                        sampled_result = list(sampled_result)
                        result.extend(sampled_result)
            # if predict segments
            else:
                preds = predicted_string
                segments = preds.split(',')
                num_frames_per_segment = int(expected_frames_num / len(segments))
                for i, segment in enumerate(segments):
                    try:
                        segment_list = segment.split(' ')
                        start_ind = segment_list.index('from') + 1
                        start = int(segment_list[start_ind])
                        start = percentage2frame(duration, start)
                        end_ind = segment_list.index('to') + 1
                        end = int(segment_list[end_ind])
                        end = percentage2frame(duration, end)

                        if i == len(segments) - 1:
                            num_frames_per_segment = expected_frames_num - len(result)

                        sampled_result = np.round(np.linspace(start, end, num_frames_per_segment)).astype(int)
                        result.extend(sampled_result)

                    except:
                        print("error ", segment)
                print(duration, preds, result)


            video_info[qid] = {"frame_indices": result}

        pkl.dump(video_info, open(os.path.join(save_dir, f"{id}.pkl"), "wb"))




