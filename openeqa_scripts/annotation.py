import os
import pandas as pd
import json
import pickle as pkl
from decord import VideoReader
import argparse
import random

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question_only", type=str, default="True", choices=["True", "False"])
    parser.add_argument("--predict_frames_number", type=int, default=6)
    parser.add_argument("--sample_size", type=int, default=-1)
    args = parser.parse_args()

    return args

if __name__ == "__main__":

    args = parse_args()

    annotation_f = "/mnt/mir/datasets/vlm-evaluation-datasets/openeqa/open_eqa_v0.csv"
    annotation_f = pd.read_csv(annotation_f)

    save_path = "/mnt/mir/datasets/vlm-evaluation-datasets/openeqa/vtimellm_annotations/"

    best_frames_dir = "/mnt/opr/fengchan/dataset/sparse_bwd/psuedo_selected_frames/open_eqa_2/"

    video_dir = "/mnt/mir/datasets/vlm-evaluation-datasets/openeqa/data/videos/"
    if args.question_only == "True":
        question_only = True 
    elif args.question_only == "False":
        question_only = False
    else:
        raise ValueError
        
    predict_frames = args.predict_frames_number

    # save train and val videos to two lists
    # videos = annotation_f.video_name.unique()
    # random.shuffle(videos)
    # train_video_len = int(len(videos)*0.7)
    # train_videos = videos[:train_video_len]
    # val_videos = videos[train_video_len:]

    train_annotation_f = "./data/open_eqa/train.csv"
    train_annotation_f = pd.read_csv(train_annotation_f)
    train_videos = train_annotation_f.video_name.unique()

    val_annotation_f = "./data/open_eqa/val.csv"
    val_annotation_f = pd.read_csv(val_annotation_f)
    val_videos = val_annotation_f.video_name.unique()


    train_json_list = []
    val_json_list = []

    for index, row in annotation_f.iterrows():
        if args.sample_size > 0:
            if index == args.sample_size: break
        info = {}
        video_name = row["video_name"]
        question = row['question']
        qid = row["question_id"]
        answer = row["answer"]
        question_type = row["question_type"]

        best_frames_f = os.path.join(best_frames_dir, str(video_name)+".pkl")
        best_frames = pkl.load(open(best_frames_f, "rb"))

        frame_indices = best_frames[qid]["frame_indices"][:predict_frames]

        vr = VideoReader(os.path.join(video_dir, str(video_name)+'.mp4'))
        fps_vid = vr.get_avg_fps()
        duration = len(vr) / fps_vid

        info["id"] = video_name
        info["qid"] = qid
        info["meta"] = {}
        info["meta"]["duration"] = duration

        tokens = {}
        for i, ind in enumerate(frame_indices):
            tokens[f"<s{i}>"] = ind
        info["meta"]['token'] = tokens
        info["conversations"] = []

        if question_only:
            question_dict = {"from": "human", "value": f"This is a video with 100 frames: <video>\n Please select the top 6 frames that can best help answer the question:\n  Question: {question}?\n The output 6 frames should be different and ranked in decreasing order of importance."}
        else:
            question_dict = {"from": "human", "value": f"This is a video with 100 frames: <video>\n Please select the top 6 frames that align most with the question and answer pair:\n  Question: {question}?\n Answer: {answer}.\n The output 6 frames should be different and ranked in decreasing order of importance."}

        info['conversations'].append(question_dict)

        qid_tokens_prompt = ' '.join(list(tokens.keys()))

        answer_dict = {"from": "gpt", "value": f"{qid_tokens_prompt}"}
        info['conversations'].append(answer_dict)

        if video_name in train_videos:
            train_json_list.append(info)
        elif video_name in val_videos:
            val_json_list.append(info)


    if args.sample_size > 0:
        train_json_list = train_json_list[:args.sample_size]

    # save train json list
    save_f = f"train-{predict_frames}-frames"
    val_save_f = f"val-{predict_frames}-frames"

    if question_only:
        save_f = save_f + "_question"
        val_save_f = val_save_f + "_question"
    else:
        save_f = save_f + "_answer"
        val_save_f = val_save_f + "_answer"

    if args.sample_size > 0:
        save_f = save_f + f"_{args.sample_size}-samples"
        val_save_f = val_save_f + f"_{args.sample_size}-samples"

    out_f = open(f"/mnt/sun/fengchan/projects-ng/xizi/VTimeLLM/data/open_eqa/{save_f}.json", "w")
    json.dump(train_json_list, out_f, indent = 6)
    out_f.close() 

    out_f = open(f"/mnt/sun/fengchan/projects-ng/xizi/VTimeLLM/data/open_eqa/{val_save_f}.json", "w")
    json.dump(val_json_list, out_f, indent = 6)
    out_f.close()
