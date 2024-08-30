import os
root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),  "..")
import sys
sys.path.append(root_dir)

import clip
import re
import argparse
import torch
import json
import numpy as np
from tqdm import tqdm
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
from vtimellm.model.builder import load_pretrained_model
from vtimellm.utils import disable_torch_init
from vtimellm.mm_utils import VideoExtractor
from vtimellm.inference import inference
import pickle as pkl
import random

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    from PIL import Image
    BICUBIC = Image.BICUBIC

def convert(duration, x):
    x = x / duration * 100
    x = str(min(round(x), 99))
    if len(x) == 1:
        x = "0" + x
    return x

def percentage2frame(duration, x):
    x = int(x / 100 * duration )
    return x

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_path", type=str, default="checkpoints/clip/ViT-L-14.pt")
    parser.add_argument("--pretrain_mm_mlp_adapter", type=str, default="checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin")
    parser.add_argument("--stage2", type=str, default="checkpoints/vtimellm-vicuna-v1-5-7b-stage2")
    parser.add_argument("--stage3", type=str, default="checkpoints/vtimellm-vicuna-v1-5-7b-stage3")
    parser.add_argument("--model_base", type=str, default="/path/to/vicuna-7b-v1.5")
    parser.add_argument("--data_path", type=str, default="vtimellm/eval/data_example.json")
    parser.add_argument("--feat_folder", type=str, default=None)
    parser.add_argument("--video_folder", type=str, default=None)
    parser.add_argument("--task", type=str, default='all', choices=['all', 'grounding', 'captioning', 'nextqa'])
    parser.add_argument("--log_path", type=str, default='vtimellm/eval/log/example_log.txt')
    parser.add_argument("--frame_or_segment", type=str, default='frame')
    parser.add_argument("--save_path", type=str, default="/home/fengchan/stor_arc/workspace/long-video-vlm/xizi/vtimellm/frame_selection_results/")
    args = parser.parse_args()
    return args

def iou(outputs, gt):
    matches = re.search(r"(\d{2}) (to|and) (\d{2})", outputs)
    if not matches:
        return 0
    from_number = float(matches.group(1)) / 100
    to_number = float(matches.group(3)) / 100
    s, e = gt
    intersection = max(0, min(to_number, e) - max(from_number, s))
    union = max(to_number, e) - min(from_number, s)
    iou = intersection / union
    return round(iou, 2)


def write_log(log_path, video_id, task, query_id, answer, info=None):
    log = {
        'video_id': video_id,
        'task': task,
        'query_id': query_id,
        'answer': answer
    }
    if info is not None:
        log['info'] = info
    with open(log_path, 'a') as f:
        f.write(json.dumps(log) + '\n')

questions = {
    'grounding': ['During which frames can we see {}?'],
    'captioning': ['Could you please describe the events in the video in detail? Be specific about the activities of individuals, their surroundings, and interactions with others. The output should be in JSON format, structured as follows: {"event": "xx", "timestamps": "from xx to xx"}.']
}

if __name__ == "__main__":
    args = parse_args()
    disable_torch_init()
    tokenizer, model, context_len = load_pretrained_model(args, args.stage2, args.stage3)
    model = model.cuda()
    model.to(torch.float16)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.video_folder is not None:
        clip_model, _ = clip.load(args.clip_path)
        clip_model.eval()
        clip_model = clip_model.cuda()

        video_loader = VideoExtractor(N=100)

        transform = Compose([
            Resize(224, interpolation=BICUBIC),
            CenterCrop(224),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    js = json.load(open(args.data_path))
    video = {}
    for data in tqdm(js):
        features = None
        id = data["id"]
        qid = data["qid"]
        qid = str(qid)
        if id not in video:
            video[id] = {}
        if id not in qid:
            qid = id + qid
        video[id][qid] = {}

        if args.feat_folder is not None:
            feat_path = os.path.join(args.feat_folder, f"{id}.npy")
            if os.path.isfile(feat_path):
                features = torch.from_numpy(np.load(feat_path)).cuda()

        if features is None and args.video_folder is not None:
            for ext in ['mp4', 'mkv', 'webm']:
                video_path = os.path.join(args.video_folder, f"{id}.{ext}")
                if os.path.isfile(video_path):
                    _, images = video_loader.extract({'id': None, 'video': video_path})

                    images = transform(images / 255.0)
                    images = images.to(torch.float16)
                    with torch.no_grad():
                        features = clip_model.encode_image(images.to('cuda'))

        if features is None:
            print(f'Can not find video {id}')
            continue
 
        if args.task in ['captioning', 'all']:
            for query_id, query in enumerate(questions['captioning']):
                answer = inference(model, features, "<video>\n " + query, tokenizer)
                write_log(args.log_path, id, 'captioning', query_id, answer)
      
        if args.task in ['grounding', 'all']:
            for sentence_id, (timestamps, sentence) in enumerate(zip(data['timestamps'], data['sentences'])):
                sentence = sentence.strip().lower()
                if sentence.endswith("."):
                    sentence = sentence[:-1]

                for query_id, query in enumerate(questions['grounding']):
                    answer = inference(model, features, "<video>\n" + query.format(sentence), tokenizer)
                    gt = (timestamps[0] / data['duration'], timestamps[1] / data['duration'])
                    u = iou(answer, gt)
                    write_log(args.log_path, id, 'grounding', query_id, answer, info={"sentence_id": sentence_id, 'iou': u})

        if args.task in ['nextqa']:
            vid_id = data["id"]
            question = data["conversations"][0]["value"]
            answer = data["conversations"][1]["value"].split(',')[-1]
            groundtruths = data["meta"]["token"]
            duration = data["meta"]["duration"]
            groundtruth = []
            for k, v in groundtruths.items():
                groundtruth.append(convert(duration, v))

            prediction = inference(model, features, question, tokenizer, do_sample=False)
            if ',' in prediction:
                predicted_frames = prediction.split(',')[0]
                predicted_answer = prediction.split(',')[-1]
                video[id][qid]['answer'] = predicted_answer
                # eliminate the From at the beginning
                predicted_frames = predicted_frames.split(' ')[1:]
            else:
                predicted_frames = prediction.split(' ')


            result = []
            for frm in predicted_frames:
                try:
                    frm = int(frm)
                    frm = percentage2frame(duration, frm)
                    result.append(frm)
                except:
                    continue
            while len(result) < 6:
                result.append(random.randint(0, int(duration)))
            print(f"psuedo best frames: {groundtruth}\tpredictions: {predicted_frames}")


            video[id][qid]['frame_indices'] = result 
            # write_log(args.log_path, id, 'grounding', query_id, answer, info={"sentence_id": sentence_id, 'iou': u})
    for vid in video.keys():
        save_data = video[vid]
        pkl.dump(save_data, open(os.path.join(args.save_path, f"{vid}.pkl"), "wb"))
