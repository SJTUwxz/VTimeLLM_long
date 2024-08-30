import os
import pandas as pd
import json
import pickle as pkl
import argparse
import random

if __name__ == "__main__":

    annotation_f = "/mnt/mir/datasets/vlm-evaluation-datasets/openeqa/open_eqa_v0.csv"
    annotation_f = pd.read_csv(annotation_f)
    
    train_f = json.load(open('data/open_eqa/train-6-frames_question.json', 'r'))
    val_f = json.load(open('data/open_eqa/val-6-frames_question.json', 'r'))

    train_vid_list = []
    val_vid_list = []

    for vid_info in train_f:
        train_vid_list.append(vid_info["id"])
    for vid_info in val_f:
        val_vid_list.append(vid_info["id"])

    print(len(set(train_vid_list)), len(set(val_vid_list)))

    mask = annotation_f['video_name'].isin(train_vid_list)
    train_csv = annotation_f[mask]
    print(len(train_csv.video_name.unique()))
    train_csv.to_csv('./data/open_eqa/train.csv', index=False)

    mask = annotation_f['video_name'].isin(val_vid_list)
    val_csv = annotation_f[mask]
    print(len(val_csv.video_name.unique()))
    val_csv.to_csv('./data/open_eqa/val.csv', index=False)
