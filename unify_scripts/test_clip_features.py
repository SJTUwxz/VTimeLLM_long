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
from PIL import Image
import requests

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
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    image = np.array(image)
    image = torch.tensor(image)
    image = image.permute(2, 0, 1)
    image = image.unsqueeze(0)

    transform = Compose([
        Resize(224, interpolation=BICUBIC),
        CenterCrop(224),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    images = transform(image / 255.0)
    images = images.to(torch.float16)
    with torch.no_grad():
        features = clip_model.encode_image(images.to('cuda')).cpu().detach()
    print(features)
