import os
import pandas as pd
import json
import pickle as pkl
from decord import VideoReader
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question_only", type=str, default="True", choices=["True", "False"])
    parser.add_argument("--train", type=str, choices=["True", "False"])
    parser.add_argument("--traintest", type=str, choices=["True", "False"], default="False")
    parser.add_argument("--sample_size", type=int, default=-1)
    args = parser.parse_args()

    return args

def frames2segment(frames, gap=15):
    # frames is a list of 6 frames
    # gap determines if one frame and the next frame has 15 frames gap, they belong to different intervals
    frames.sort()
    groups = []
    prev = 0
    curr_group = []
    for i, frm in enumerate(frames):
        
        if i == 0:
            prev = frm
            curr_group.append(frm)
            continue
        if frm - curr_group[-1] > gap:
            if len(curr_group) == 1:
                curr_group.append(curr_group[0] + 1)
            groups.append([curr_group[0], curr_group[-1]])
            curr_group = [frm]
        else:
            curr_group.append(frm)
    if curr_group:
        if len(curr_group) == 1:
            curr_group.append(curr_group[0] + 1)
        groups.append([curr_group[0], curr_group[-1]])
    return groups


if __name__ == "__main__":

    args = parse_args()

    if args.question_only == "True":
        question_only = True 
    elif args.question_only == "False":
        question_only = False
    else:
        raise ValueError

    if args.train == "True":
        if args.traintest == "True":
            dset = f"next_qa_traintest_segment"
        else:
            dset = f"next_qa_train_segment"
    else:
        dset = f"next_qa_segment"

    if question_only:
        dset = dset + "_question"
    else:
        dset = dset + "_answer"

    if args.sample_size > 0:
        dset = dset + f"_{args.sample_size}"

    nextqa_annotation_dir = "/mnt/mir/datasets/vlm-evaluation-datasets/nextqa/annotations/"

    print("output file: ", dset)

            
    if "train" in dset:
        if "traintest" in dset:
            nextqa_annotation = pd.read_csv(os.path.join("/mnt/mir/datasets/vlm-evaluation-datasets/nextqa/annotations/train_subset_igvlm_eval.csv"))
        else:
            nextqa_annotation = pd.read_csv(os.path.join(nextqa_annotation_dir, "train.csv"))
        best_frames_dir = os.path.join("/home/fengchan/stor/dataset/sparse_bwd/psuedo_selected_frames/", "next_qa_train")
    else:
        nextqa_annotation = pd.read_csv(os.path.join(nextqa_annotation_dir, "val.csv"))
        best_frames_dir = os.path.join("/home/fengchan/stor/dataset/sparse_bwd/psuedo_selected_frames/", "next_qa")

    video_dir = "/mnt/mir/datasets/vlm-evaluation-datasets/nextqa/videos/"

    json_list = []

    for index, row in nextqa_annotation.iterrows():
        if args.sample_size > 0:
            if index == args.sample_size: break
        info = {}
        if "traintest" in dset:
            video_name = row["video_name"]
        else:
            video_name = row['video']
        frame_count = row['frame_count']
        question = row['question']
        options = [row["a0"], row["a1"], row["a2"], row["a3"], row["a4"]]
        if "traintest" in dset:
            answer = row["answer"]
            qid = row["question_id"]
        else:
            answer_number = int(row['answer'])
            answer = options[answer_number]
            qid = row['qid']

        best_frames_f = os.path.join(best_frames_dir, str(video_name)+".pkl")
        best_frames = pkl.load(open(best_frames_f, "rb"))
        if "train" not in dset:
            qid = str(video_name) + str(qid)
        if "traintest" in dset:
            qid = str(qid)[len(str(video_name)):]
        qid = int(qid)
        frame_indices = best_frames[qid]["frame_indices"]
        segments = frames2segment(frame_indices)

        vr = VideoReader(os.path.join(video_dir, str(video_name)+'.mp4'))
        fps_vid = vr.get_avg_fps()
        duration = float(frame_count) / fps_vid

        info["id"] = str(video_name)
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

        json_list.append(info)

# out_f = open("/home/fengchan/projects-ng/xizi/VTimeLLM/data/nextqa_train.json", "w")
    out_f = open(f"/mnt/sun/fengchan/projects-ng/xizi/VTimeLLM/data/{dset}.json", "w")
    json.dump(json_list, out_f, indent = 6)

    out_f.close()
