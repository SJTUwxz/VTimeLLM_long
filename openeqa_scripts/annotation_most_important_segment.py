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
    parser.add_argument("--sample_size", type=int, default=-1)
    args = parser.parse_args()

    return args

def frames2segment(duration, frames, segment_nearby=2):
    # frames is a list of 6 frames
    # gap determines if one frame and the next frame has 15 frames gap, they belong to different intervals
    
    best_frame = frames[0]
    last_frame = max(frames)
    frames.sort()
    segments = []
    curr_group = []
    for i, frm in enumerate(frames):
        if i == 0:
            curr_group.append(frm)
            continue
        if frm - curr_group[-1] <= segment_nearby:
            curr_group.append(frm)
        else:
            segments.append(curr_group)
            curr_group = [frm]
    if len(curr_group) > 0:
        segments.append(curr_group)


    for segment in segments:
        start = segment[0]
        end = segment[-1]
        if best_frame >= start and best_frame <= end:
            return [[start, end]]
    return None


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

        vr = VideoReader(os.path.join(video_dir, str(video_name)+'.mp4'))
        fps_vid = vr.get_avg_fps()
        duration = len(vr) / fps_vid

        frame_indices = best_frames[qid]["frame_indices"]
        segments = frames2segment(duration, frame_indices)
        print(duration,frame_indices, segments)


        info["id"] = video_name
        info["qid"] = qid
        info["meta"] = {}
        info["meta"]["duration"] = duration

        tokens = {}
        ind2token = {}
        idx = 0
        for start, end in segments:
            tokens[f"<s{idx}>"] = start
            ind2token[start] = f"<s{idx}>"
            idx = idx + 1
            tokens[f"<e{idx}>"] = end 
            ind2token[end] = f"<e{idx}>"
            idx = idx + 1

        info["meta"]['token'] = tokens
        info["conversations"] = []

        segment_prompt = []
        for start, end in segments:
            segment_prompt.append(f" from {ind2token[start]} to {ind2token[end]}")


        if question_only:
            question_dict = {"from": "human", "value": f"This is a video with 100 frames: <video>\n Please select the top 6 frames that can best help answer the question:\n  Question: {question}?\n The output 6 frames should be ranked in decreasing order of importance."}
        else:
            question_dict = {"from": "human", "value": f"This is a video with 100 frames: <video>\n Please select the top 6 frames that align most with the question and answer pair:\n  Question: {question}?\n Answer: {answer}.\n The output 6 frames should be ranked in decreasing order of importance."}
        info['conversations'].append(question_dict)

        # qid_tokens_prompt = ' '.join(list(tokens.keys()))
        qid_tokens_prompt = ','.join(segment_prompt)

        answer_dict = {"from": "gpt", "value": f"{qid_tokens_prompt}"}
        info['conversations'].append(answer_dict)

        if video_name in train_videos:
            train_json_list.append(info)
        elif video_name in val_videos:
            val_json_list.append(info)


    if args.sample_size > 0:
        train_json_list = train_json_list[:args.sample_size]

    # save train json list
    save_f = f"train-bestsegment"
    val_save_f = f"val-bestsegment"

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
