import argparse
import os
from multiprocessing import Pool
import pickle as pkl

parser = argparse.ArgumentParser()
parser.add_argument("--source_path", required=True, type=str)
parser.add_argument("--video_path", required=True, type=str)
args = parser.parse_args()

source_path = args.source_path
video_path = args.video_path

import copy
import json

import numpy as np
from tqdm import tqdm

print("Loading data.")

with open(source_path, "r") as f:
    packed_data = json.load(f)

print("Start downloading.")

video_names = pkl.load(open("./download_ids.pkl", "rb"))
exist_ids = pkl.load(open("./exist_ids.pkl", "rb"))
to_download = [x for x in video_names if x not in exist_ids]
#exist_videos=[]
video_names=video_names[::-1]

youtube_video_format = "https://www.youtube.com/watch?v={}"
video_path_format = os.path.join(video_path, "{}.mp4")

def download(video_name):
    try:
        url = youtube_video_format.format(video_name)
        file_path = video_path_format.format(video_name)
        if os.path.exists(file_path):
            return
        os.system("yt-dlp --username oauth2 --password '' -o " + file_path + " -f 134 " + url)
        print(f"Downloading of Video {video_name} has finished.")
    except:
        print(f"Downloading of Video {video_name} has failed.")

pool = Pool(16)
pool.map(download, to_download)
# for video_name in video_names:
print("Finished.")

