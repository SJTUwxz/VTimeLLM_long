import os
root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
import sys
sys.path.append(root_dir)
import clip
import json
from tqdm import tqdm
from vtimellm.mm_utils import VideoExtractor
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
import torch
from glob import glob

import numpy as np
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    from PIL import Image
    BICUBIC = Image.BICUBIC

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_folder", type=str, default="/mnt/mir/datasets/vlm-evaluation-datasets/nextqa/videos")
    parser.add_argument("--save_dir", type=str, default="/mnt/mir/datasets/vlm-evaluation-datasets/nextqa/clip_features/")
    parser.add_argument("--num_features", type=int, default=100)
    args = parser.parse_args()

    return args

if __name__ == "__main__":

    args = parse_args()

    video_folder = args.video_folder
    clip_path = "./checkpoints/clip/ViT-L-14.pt"

    clip_model, _ = clip.load(clip_path)
    clip_model.eval()
    clip_model = clip_model.cuda()

    save_dir = args.save_dir


    video_loader = VideoExtractor(N=args.num_features)

    transform = Compose([
        Resize(224, interpolation=BICUBIC),
        CenterCrop(224),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    for filename in glob(f'{video_folder}/*.mp4', recursive=True):
        id = os.path.basename(filename)[:-4]

        features = None
        save_npy = os.path.join(save_dir, str(id) + ".npy")
        if os.path.exists(save_npy):
            continue
        else:
            open(save_npy, 'w').close()

        video_path = os.path.join(video_folder, f"{id}.mp4")
        if os.path.isfile(video_path):
            _, images = video_loader.extract({'id': None, 'video': video_path})

            images = transform(images / 255.0)
            images = images.to(torch.float16)
            with torch.no_grad():
                features = clip_model.encode_image(images.to('cuda')).cpu().detach()
        # features shape 100, 768
        assert (features is not None), f"Cannot find video {id}"
        np.save(save_npy, features)

