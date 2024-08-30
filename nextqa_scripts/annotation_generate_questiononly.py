import os
import pandas as pd
import json
import pickle as pkl
from decord import VideoReader

nextqa_annotation_dir = "/mnt/mir/datasets/vlm-evaluation-datasets/nextqa/annotations/"
dset = "next_qa_train_question"

if "train" in dset:
    nextqa_annotation = pd.read_csv(os.path.join(nextqa_annotation_dir, "train.csv"))
else:
    nextqa_annotation = pd.read_csv(os.path.join(nextqa_annotation_dir, "val.csv"))

if "train" in dset:
    best_frames_dir = os.path.join("/home/fengchan/stor/dataset/sparse_bwd/psuedo_selected_frames/", "next_qa_train")
else:
    best_frames_dir = os.path.join("/home/fengchan/stor/dataset/sparse_bwd/psuedo_selected_frames/", "next_qa")

video_dir = "/mnt/mir/datasets/vlm-evaluation-datasets/nextqa/videos/"

json_list = []

for index, row in nextqa_annotation.iterrows():
    info = {}
    video_name = row['video']
    frame_count = row['frame_count']
    question = row['question']
    options = [row["a0"], row["a1"], row["a2"], row["a3"], row["a4"]]
    answer_number = int(row['answer'])
    answer = options[answer_number]
    qid = row['qid']

    best_frames_f = os.path.join(best_frames_dir, str(video_name)+".pkl")
    best_frames = pkl.load(open(best_frames_f, "rb"))
    if "train" not in dset:
        qid = str(video_name) + str(qid)
    qid = int(qid)
    frame_indices = best_frames[qid]["frame_indices"]

    vr = VideoReader(os.path.join(video_dir, str(video_name)+'.mp4'))
    fps_vid = vr.get_avg_fps()
    duration = float(frame_count) / fps_vid

    info["id"] = str(video_name)
    info["meta"] = {}
    info["meta"]["duration"] = duration

    tokens = {}
    for i, ind in enumerate(frame_indices):
        tokens[f"<s{i}>"] = ind
    info["meta"]['token'] = tokens
    info["conversations"] = []

    question_dict = {"from": "human", "value": f"This is a video with 100 frames: <video>\n Please select the top 6 frames that can best help answer the question:\n  Question: {question}?\n The output 6 frames should be ranked in decreasing order of importance."}
    info['conversations'].append(question_dict)

    qid_tokens_prompt = ' '.join(list(tokens.keys()))

    answer_dict = {"from": "gpt", "value": f"From {qid_tokens_prompt}, {answer}"}
    info['conversations'].append(answer_dict)

    json_list.append(info)

# out_f = open("/home/fengchan/projects-ng/xizi/VTimeLLM/data/nextqa_train.json", "w")
out_f = open(f"/mnt/sun/fengchan/projects-ng/xizi/VTimeLLM/data/{dset}.json", "w")
json.dump(json_list, out_f, indent = 6)

out_f.close()



