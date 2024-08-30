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

def convert(duration, x, num_features_per_video):
    x = x / duration * num_features_per_video 
    x = str(min(round(x), (num_features_per_video-1)))
    if num_features_per_video > 100:
        if len(x) == 1:
            x = "00" + x
        elif len(x) == 2:
            x = "0" + x
    else:
        if len(x) == 1:
            x = "0" + x
    return x

def percentage2frame(duration, x, num_features_per_video):
    x = int(x / num_features_per_video * duration )
    return x

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_path", type=str, default="checkpoints/clip/ViT-L-14.pt")
    parser.add_argument("--pretrain_mm_mlp_adapter", type=str, default="checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin")
    parser.add_argument("--stage2", type=str, default="checkpoints/vtimellm-vicuna-v1-5-7b-stage2")
    parser.add_argument("--stage3", type=str, default="checkpoints/vtimellm-vicuna-v1-5-7b-stage3")
    parser.add_argument("--model_base", type=str, default="checkpoints/vicuna-7b-v1.5")
    parser.add_argument("--data_path", type=str, default="vtimellm/eval/data_example.json")
    parser.add_argument("--feat_folder", type=str, default=None)
    parser.add_argument("--num_features_per_video", type=int, default=100)
    parser.add_argument("--video_folder", type=str, default=None)
    parser.add_argument("--task", type=str, default='custom', choices=['all', 'grounding', 'captioning', 'custom'])
    parser.add_argument("--log_path", type=str, default='vtimellm/eval/log/example_log.txt')
    parser.add_argument("--frame_or_segment", type=str, default='frame')
    parser.add_argument("--save_path", type=str, default="/mnt/opr/fengchan/dataset/sparse_bwd/predicted_best_frames/")
    parser.add_argument("--missing_prediction_complete", type=str, default='uniform', choices=["uniform", "reuse"])
    parser.add_argument(
        "--predict_segment",
        action="store_true",
        help="whether to use selected chunks",
    )
    parser.add_argument(
        "--frames_num",
        type=int,
        help="whether to use selected chunks",
    )
    parser.add_argument(
        "--question_or_answer",
        type=str,
        choices=["question", "answer"]
    )
    parser.add_argument(
        "--subset",
        type=str,
        choices=["traintest", "val", "train"]
    )
    parser.add_argument(
        "--dset",
        type=str,
        choices=["next_qa", "openeqa"]
    )
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

    save_path = args.save_path
    dset = args.dset
    frames_num = args.frames_num
    question_or_answer = args.question_or_answer
    subset = args.subset
    missing_prediction_complete = args.missing_prediction_complete
    num_features_per_video = args.num_features_per_video
    print("MISS PREDICTION COMPLETE: ", missing_prediction_complete)

    if args.dset== "next_qa":
        if subset == "traintest":
            dset = dset + "_traintest"
        dataset = dset

        if args.predict_segment:
            dset = dset + f"_segment_{question_or_answer}"
            save_path = os.path.join("/home/fengchan/stor/dataset/sparse_bwd/predicted_best_frames/", dataset, f"{subset}_segment_{question_or_answer}")
            os.makedirs(save_path, exist_ok=True)
        else:
            dset = dset + f"_{frames_num}-frame_{question_or_answer}"
            save_path = os.path.join("/home/fengchan/stor/dataset/sparse_bwd/predicted_best_frames/", dataset, f"{subset}_{frames_num}-frame_{question_or_answer}")
            os.makedirs(save_path, exist_ok=True)

    elif args.dset == "openeqa":
        # if subset == "train":
        #     dset = dset + "_train"
        # elif subset == "val":
        #     dset = dset + "_val"
        # dataset = dset
        #
        # if args.predict_segment:
        #     dset = dset + f"_segment_{question_or_answer}"
        #     save_path = os.path.join("/home/fengchan/stor/dataset/sparse_bwd/predicted_best_frames/", dataset, f"{subset}_segment_{question_or_answer}")
        #     os.makedirs(save_path, exist_ok=True)
        #     os.makedirs(os.path.join(save_path, "hm3d-v0"), exist_ok=True)
        #     os.makedirs(os.path.join(save_path, "scannet-v0"), exist_ok=True)
        # else:
        #     dset = dset + f"_{frames_num}-frame_{question_or_answer}"
        #     save_path = os.path.join("/home/fengchan/stor/dataset/sparse_bwd/predicted_best_frames/", dataset, f"{subset}_{frames_num}-frame_{question_or_answer}")
        #     os.makedirs(save_path, exist_ok=True)
        #     os.makedirs(os.path.join(save_path, "hm3d-v0"), exist_ok=True)
        #     os.makedirs(os.path.join(save_path, "scannet-v0"), exist_ok=True)
        save_path = args.save_path


    if args.video_folder is not None:
        clip_model, _ = clip.load(args.clip_path)
        clip_model.eval()
        clip_model = clip_model.cuda()

        video_loader = VideoExtractor(N=num_features_per_video)

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
        if id not in qid and args.dset=="next_qa":
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

        if args.task in ['custom']:
            vid_id = data["id"]
            question = data["conversations"][0]["value"]
            answer = data["conversations"][1]["value"].split(',')[-1]
            groundtruths = data["meta"]["token"]
            duration = data["meta"]["duration"]
            groundtruth = []
            for k, v in groundtruths.items():
                groundtruth.append(convert(duration, v, num_features_per_video))

            prediction = inference(model, features, question, tokenizer, do_sample=False)
            result = []
            if not args.predict_segment:
                if ',' in prediction:
                    predicted_frames = prediction.split(',')[0]
                    predicted_answer = prediction.split(',')[-1]
                    video[id][qid]['answer'] = predicted_answer
                    # eliminate the From at the beginning
                    predicted_frames = predicted_frames.split(' ')[1:]
                else:
                    predicted_frames = prediction.split(' ')


                for frm in predicted_frames:
                    try:
                        frm = int(frm)
                        frm = percentage2frame(duration, frm, num_features_per_video)
                        result.append(frm)
                    except:
                        continue
                if missing_prediction_complete == "reuse":
                    if len(result) < 6:
                        try:
                            sampled_result = random.sample(result, 6-len(result))
                            result.extend(sampled_result)
                        except:
                            missing_prediction_complete = "uniform"
                if missing_prediction_complete == "uniform":
                    missing_frames_num = 6 - len(result)
                    if len(result) < 6:
                        sampled_result = np.round(np.linspace(0, duration - 1, missing_frames_num)).astype(int)
                        sampled_result = list(sampled_result)
                        result.extend(sampled_result)
            else:
                preds = prediction
                segments = preds.split(',')
                num_frames_per_segment = int(6 / len(segments))
                for i, segment in enumerate(segments):
                    try:
                        segment_list = segment.split(' ')
                        start_ind = segment_list.index('from') + 1
                        start = int(segment_list[start_ind])
                        start = percentage2frame(duration, start)
                        end_ind = segment_list.index('to') + 1
                        end = int(segment_list[end_ind])
                        end = percentage2frame(duration, end)

                        if i == len(segments) - 1:
                            num_frames_per_segment = 6 - len(result)

                        sampled_result = np.round(np.linspace(start, end, num_frames_per_segment)).astype(int)
                        result.extend(sampled_result)

                    except:
                        print("error ", segment)


            assert len(result) == 6, f"not enough frames in result: {prediction}"

            print(f"psuedo best frames: {groundtruth}\tpredictions: {prediction}")

            video[id][qid]['frame_indices'] = result 
            # write_log(args.log_path, id, 'grounding', query_id, answer, info={"sentence_id": sentence_id, 'iou': u})
    for vid in video.keys():
        save_data = video[vid]
        pkl.dump(save_data, open(os.path.join(save_path, f"{vid}.pkl"), "wb"))

