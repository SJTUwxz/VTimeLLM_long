import os
import sys
import argparse
import torch
from vtimellm.constants import IMAGE_TOKEN_INDEX, TEMPORAL_TOKEN_INDEX, SEG_START, SEG_END
from vtimellm.conversation import conv_templates, SeparatorStyle
from vtimellm.model.builder import load_pretrained_model, load_lora
from vtimellm.utils import disable_torch_init
from vtimellm.mm_utils import tokenizer_image_token, tokenizer_image_segment_token, KeywordsStoppingCriteria, VideoExtractor
from PIL import Image
import requests
from io import BytesIO
from transformers import TextStreamer
from easydict import EasyDict as edict
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    from PIL import Image
    BICUBIC = Image.BICUBIC
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
import numpy as np
import clip


def inference(model, image, query, tokenizer, do_sample=False):
    conv = conv_templates["v1"].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image[None,].cuda(),
            do_sample=do_sample,
            temperature=0.05,
            num_beams=1,
            # no_repeat_ngram_size=3,
            max_new_tokens=1024,
            use_cache=True)

        # https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py#L1295

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    return outputs

def temporal_segment_inference(model, image, query, tokenizer,  args, do_sample=False):
    conv = conv_templates["v1"].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # can also use the above inference but we want to test out replacing generate with forward
    # if not args.temporal_loss:
    #     input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    #
    #     stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    #     keywords = [stop_str]
    #     stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    #
    #     # original generate function
    #     with torch.inference_mode():
    #         output_ids = model.generate(
    #             input_ids,
    #             images=image[None,].cuda(),
    #             do_sample=do_sample,
    #             temperature=0.05,
    #             num_beams=1,
    #             # no_repeat_ngram_size=3,
    #             max_new_tokens=1024,
    #             use_cache=True)
    #
    #         # https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py#L1295
    #
    #     input_token_len = input_ids.shape[1]
    #     n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    #     if n_diff_input_output > 0:
    #         print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    #     outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    #     outputs = outputs.strip()
    #     if outputs.endswith(stop_str):
    #         outputs = outputs[:-len(stop_str)]
    #     outputs = outputs.strip()
    #
    #     with torch.inference_mode():
    #         cached = None
    #         inputs = input_ids.clone()
    #         for i in range(20): # 20 as the max length here, as an ex
    #             if cached is None:
    #                 outputs = model(input_ids=inputs, images=image[None,].cuda(), return_dict=True, output_hidden_states=True, use_cache=True)
    #                 cached = outputs.past_key_values
    #             else:
    #                 outputs = model(input_ids=pred_id, images=image[None,].cuda(), past_key_values=cached, return_dict=True, output_hidden_states=True, use_cache=True, is_generate=True)
    #                 cached = outputs.past_key_values
    #             pred_id = torch.argmax(outputs.logits[0, -1]).unsqueeze(0).unsqueeze(0)
    #             inputs = torch.hstack((inputs, pred_id))
    #
    #         output_ids = inputs
    #
    #     input_token_len = input_ids.shape[1]
    #     # make sure the input ids and the first elements of the output ids are the same
    #     n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    #     if n_diff_input_output > 0:
    #         print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    #     outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    #     outputs = outputs.strip()
    #     if outputs.endswith(stop_str):
    #         outputs = outputs[:-len(stop_str)]
    #     outputs = outputs.strip()
    #     return outputs
    if args.temporal_loss:
        input_ids = tokenizer_image_segment_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, TEMPORAL_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            (
                _,
                _,
                _,
                _,
                inputs_embeds,
                _, 
            ) = model.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids=None,
                attention_mask=None,
                past_key_values=None,
                labels=None,
                images=image[None,].cuda(),
            )
            
            seg_start = None
            cached = None
            next_token_embedding=None
            print(model.model.temporal_projector.weight)
            for i in range(5): # 20 as the max length here, as an ex
                if cached is None:
                    outputs = model(inputs_embeds=inputs_embeds, images=image[None,].cuda(), return_dict=True, output_hidden_states=True, use_cache=True)
                    cached = outputs.past_key_values
                else:
                    outputs = model(inputs_embeds=next_token_embedding, images=image[None,].cuda(), past_key_values=cached, return_dict=True, output_hidden_states=True, use_cache=True, is_generate=True)
                    cached = outputs.past_key_values

                pred_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
                # output_ids = torch.cat((output_ids, pred_id.unsqueeze(0)), dim=1)
                if tokenizer.convert_ids_to_tokens(int(pred_id.item())) == SEG_START and seg_start is None:
                    seg_start = i
                    next_token_embedding = model.model.embed_tokens(pred_id.unsqueeze(0))
                    continue

                if seg_start is not None and i == seg_start + 1:
                    seg_start_embedding = outputs.hidden_states[-1]
                    seg_start_value = model.segment_head(seg_start_embedding)

                    if args.projector_type == "simple_linear":
                        temporal_features = model.get_model().temporal_projector(seg_start_value)
                    elif args.projector_type == "angular":
                        seg_start_value = seg_start_value.to(torch.bfloat16)
                        temporal_features = model.get_model().func[0](seg_start_value)
                        temporal_features = temporal_features.to(torch.float16)
                        temporal_features = model.get_model().temporal_projector(temporal_features)
                    else:
                        raise NotImplementedError("projector_type not implemented")
                    next_token_embedding = temporal_features
                    continue

                elif seg_start is not None and i == seg_start + 2:
                    seg_end_embedding = outputs.hidden_states[-1]
                    seg_end_value = model.segment_head(seg_end_embedding)

                    if args.projector_type == "simple_linear":
                        temporal_features = model.get_model().temporal_projector(seg_end_value)
                    elif args.projector_type == "angular":
                        seg_end_value = seg_end_value.to(torch.bfloat16)
                        temporal_features = model.get_model().func[0](seg_end_value)
                        temporal_features = temporal_features.to(torch.float16)
                        temporal_features = model.get_model().temporal_projector(temporal_features)
                        # temporal_features = temporal_features.to(torch.float16)
                        # seg_end_value = seg_end_value.to(torch.float16)
                    else:
                        raise NotImplementedError("projector_type not implemented")
                    next_token_embedding = temporal_features
                    print(seg_start, seg_start_value, seg_end_value, tokenizer.convert_ids_to_tokens(int(pred_id.item())), flush=True)
                    break
                if next_token_embedding is None:
                    next_token_embedding = model.model.embed_tokens(pred_id.unsqueeze(0))

        outputs = [seg_start_value.cpu().detach().item(), seg_end_value.cpu().detach().item()]
        return outputs




def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--clip_path", type=str, default="checkpoints/clip/ViT-L-14.pt")
    parser.add_argument("--model_base", type=str, default="/path/to/vicuna-7b-v1.5")
    parser.add_argument("--pretrain_mm_mlp_adapter", type=str, default="checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin")
    parser.add_argument("--stage2", type=str, default="checkpoints/vtimellm-vicuna-v1-5-7b-stage2")
    parser.add_argument("--stage3", type=str, default="checkpoints/vtimellm-vicuna-v1-5-7b-stage3")
    parser.add_argument("--video_path", type=str, default="images/demo.mp4")
    parser.add_argument("--num_features_per_video", type=int, default=100)
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    disable_torch_init()
    tokenizer, model, context_len = load_pretrained_model(args, args.stage2, args.stage3)
    model = model.cuda()
    # model.get_model().mm_projector.to(torch.float16)
    model.to(torch.float16)
    num_features_per_video = args.num_features_per_video

    clip_model, _ = clip.load(args.clip_path)
    clip_model.eval()
    clip_model = clip_model.cuda()

    video_loader = VideoExtractor(N=num_features_per_video)
    _, images = video_loader.extract({'id': None, 'video': args.video_path})

    transform = Compose([
        Resize(224, interpolation=BICUBIC),
        CenterCrop(224),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    # print(images.shape) # <N, 3, H, W>
    images = transform(images / 255.0)
    images = images.to(torch.float16)
    with torch.no_grad():
        features = clip_model.encode_image(images.to('cuda'))

    query = "describe the video."
    print("query: ", query)
    print("answer: ", inference(model, features, "<video>\n " + query, tokenizer))

