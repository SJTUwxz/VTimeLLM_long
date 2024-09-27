import os
root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
import sys
sys.path.append(root_dir)
# import clip
import json
from tqdm import tqdm
from vtimellm.mm_utils import VideoExtractor
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
import torch
from glob import glob
from transformers import AutoProcessor, CLIPVisionModel, CLIPVisionModelWithProjection
from transformers import CLIPProcessor, CLIPModel

import numpy as np
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    from PIL import Image
    BICUBIC = Image.BICUBIC

import argparse
import datetime
import time
import pickle as pkl
from torch import nn

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_folder", type=str, default="/mnt/mir/datasets/vlm-evaluation-datasets/nextqa/videos")
    parser.add_argument("--save_dir", type=str, default="/mnt/mir/datasets/vlm-evaluation-datasets/nextqa/clip_features/")
    parser.add_argument("--num_features", type=int, default=100)
    parser.add_argument("--data_path", type=str)

    args = parser.parse_args()

    return args

if __name__ == "__main__":

    args = parse_args()

    video_folder = args.video_folder

    save_dir = args.save_dir


    model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model = model.cuda()

    video_loader = VideoExtractor(N=args.num_features)

    
    js = json.load(open(args.data_path))
    for data in tqdm(js):
        if type(data) == str:
            id = data
        else:
            id = data["id"]

        features = None
        save_npy = os.path.join(save_dir, str(id) + ".npy")
        if os.path.exists(save_npy):
            continue
        #     try:
        #         feature = np.load(save_npy, allow_pickle=True)
        #         continue
        #     except:
        #         open(save_npy, 'w').close()
        # else:
        #     open(save_npy, 'w').close()

        m = nn.AdaptiveAvgPool2d((8, 8)).cuda()

        video_path = os.path.join(video_folder, id+".mp4") 
        if os.path.isfile(video_path):
            _, images = video_loader.extract({'id': None, 'video': video_path})

            transformer_clip_images = images
            # input images shape: 100, 3, h, w
            try:
                inputs = processor(images=transformer_clip_images, return_tensors="pt", padding=True)
            except:
                continue
            inputs = inputs["pixel_values"]
            inputs = inputs.cuda()

            # inputs: 100, 3, 224, 224
            split_num = 4
            split_size = int(100/split_num)
            for i in range(4): 
                part_inputs = inputs[split_size*i:split_size*(i+1), :, :, :]
                part_outputs = model(part_inputs, return_dict=True)

                hidden_states = part_outputs.last_hidden_state

                num_images, num_tokens, hidden_size = hidden_states.shape
                pooled_output = hidden_states[:, 0:1, :]
                patch_features = hidden_states[:, 1:, :].permute(0, 2, 1).reshape(num_images, hidden_size, 16, 16)
                patch_features = m(patch_features)
                patch_features = patch_features.reshape(num_images, hidden_size, 64).permute(0, 2, 1)
                image_features =  torch.cat([pooled_output, patch_features], dim=1)

                last_hidden_state = image_features.cpu().detach()
                image_embeds = part_outputs.image_embeds.cpu().detach()

                # last_hidden_state: num_images, 257, 1024
                # image_embeds: num_images, 768
                if i == 0:
                    prev_last_hidden_state = last_hidden_state
                    prev_image_embeds = image_embeds

                else:
                    prev_last_hidden_state = torch.cat((prev_last_hidden_state, last_hidden_state), dim=0)
                    prev_image_embeds = torch.cat((prev_image_embeds, image_embeds), dim=0)



            features = {'last_hidden_state': prev_last_hidden_state, 'image_embeds': prev_image_embeds}

        # assert (features is not None), f"Cannot find video {id}"
            np.save(save_npy, features)

