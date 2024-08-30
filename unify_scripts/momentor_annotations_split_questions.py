import os
root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
import sys
sys.path.append(root_dir)
import json
from decord import VideoReader
from tqdm import tqdm
import random
import argparse

dense_captioning_templates = ["Could you please detail the events that took place during different time segments in the video?", 
                              "I’m curious about what happened at different points in the video. Could you please describe the events?",
                              "Could you provide a summary of the incidents that occurred at various timestamps in the video?",
                              "I'd like to know what events transpired during specific time intervals in the video. Could you please elaborate?",
                              "Can you give me a breakdown of the occurrences at different time stamps in the video?",
                              "I’m interested in understanding the events that unfolded at different points in the video. Could you please specify?",
                              "Could you outline the incidents that happened during different time periods in the video?",
                              "I’m trying to grasp the sequence of events in the video. Could you please outline what happened at different times?",
                              "Can you go through the video and describe what took place at different time intervals?",
                              "I’d appreciate it if you could provide a detailed account of the events that occurred at different timestamps in the video.",
                            ]
event_caption_templates = ["Can you describe what occurred from [S] to [E] in the video?",
                           "Could you tell me what happened from [S] to [E] in the video?",
                           "What transpired from [S] to [E] in the video?",
                           "Describe what took place from [S] to [E] in the video.",
                           "Tell me about the events from [S] to [E] in the video.",
                           "What was going on from [S] to [E] in the video?",
                           "Please recount what occurred from [S] to [E] in the video.",
                           "Explain what happened from [S] to [E] in the video.",
                           "Provide details about the events from [S] to [E] in the video.",
                           "Share what transpired from [S] to [E] in the video."
                           ]
temporal_grounding_templates = ["During which frames can we see [T] happening in the video?",
                                "Between which frames is [T] visible in the video?",
                                "At what point in the video can we observe [T] taking place?",
                                "Between which two frames can we witness [T] occurring in the video?",
                                "During which frames in the video can we observe [T] happening?",
                                "At which time interval in the video can we see [T] occurring?",
                                "Between which frames can we find [T] taking place in the video?",
                                "At what point in the video can we witness [T] happening?",
                                "Between which two frames in the video can we observe [T] taking place?",
                                "During which frames does [T] occur in the video?"
                                ]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_questions", type=int, default=6)
    args = parser.parse_args()

    return args

if __name__ == "__main__":

    args = parse_args()

    video_folder = "/mnt/mir/datasets/vlm-datasets/momentor-10M/videos"
    annotation = json.load(open('/mnt/mir/datasets/vlm-datasets/momentor-10M/annotations/GESM_data.format.json','r'))

    split_questions = args.split_questions

    output_json = f"./data/vtimellm_train/momentor_{split_questions}_questions.json"
    if os.path.exists(output_json):
        print("file already exists!")
        sys.exit(0)
    out_f = open(output_json, "w")
    conversation = {}

    for vid in tqdm(list(annotation.keys())):

        try:
            vr = VideoReader(os.path.join(video_folder, str(vid)+'.mp4'))     
            fps_vid = vr.get_avg_fps()
            duration = float(len(vr)) / fps_vid
        except:
            continue


        captions = annotation[vid]['captions']
        timestamps = annotation[vid]['timestamps']


        # Following VTimeLLM stage2: prob > 0.2 multiple QA, prob < 0.2 single QA
        prob = random.random()
        if prob <= 0.2:
            tokens = {}
            dialogues = []
            for i, ind in enumerate(timestamps):
                start, end = ind
                tokens[f"<s{i}>"] = start
                tokens[f"<e{i}>"] = end
            question = random.choice(dense_captioning_templates)
            answer = [] 
            for i, cap in enumerate(captions):
                if cap[-1] == ".":
                    cap = cap[:-1]
                answer.append(f"{cap}, from <s{i}> to <e{i}>.")
            answer = " ".join(answer)
            dialogues.append({"from": "human", "value": f"<video>\n{question}"})
            dialogues.append({"from": "gpt", "value": answer})

            if vid not in conversation.keys():
                conversation[vid] ={}
            conversation[vid]["id"] = vid
            conversation[vid]["meta"] = {"duration": duration}
            conversation[vid]["meta"]['token'] = tokens
            conversation[vid]["conversations"] = dialogues
            conversation[vid]["source"] = "momentor"
            

        else:
            # randomly shuffle the order of the questions so they do not occur in the order of event's occurrences
            c = list(zip(captions, timestamps))
            random.shuffle(c)
            if len(c) == 0:
                conversation.pop(vid, None)
                continue
            
            for j in range(0, len(c), split_questions): 

                splitted_c = c[j:j + split_questions]
                tokens = {}
                dialogues = []

                captions, timestamps = zip(*splitted_c)

                for i, ind in enumerate(timestamps):
                    start, end = ind
                    tokens[f"<s{i}>"] = start
                    tokens[f"<e{i}>"] = end

                for i, cap in enumerate(captions):
                    prob = random.random()
                    # if prob < 0.5, get list of event captioning questions
                    if prob < 0.5:
                        question = random.choice(event_caption_templates)
                        question = question.replace("[S]", f"<s{i}>")
                        question = question.replace("[E]", f"<e{i}>")

                        answer = cap

                    else:
                        question = random.choice(temporal_grounding_templates)
                        if cap[-1] == ".":
                            cap = cap[:-1]
                        cap = cap.lower()
                        question = question.replace("[T]", cap)

                        answer = f"From <s{i}> to <e{i}>."

                    if i == 0:
                        dialogues.append({"from": "human", "value": f"<video>\n{question}"})
                    else:
                        dialogues.append({"from": "human", "value": f"{question}"})

                    dialogues.append({"from": "gpt", "value": answer})
            
                if vid+str(j) not in conversation.keys():
                    conversation[vid+str(j)] ={}
                conversation[vid+str(j)]["id"] = vid
                conversation[vid+str(j)]["meta"] = {"duration": duration}
                conversation[vid+str(j)]["meta"]['token'] = tokens
                conversation[vid+str(j)]["conversations"] = dialogues
                conversation[vid+str(j)]["source"] = "momentor"

    json.dump(list(conversation.values()), out_f, indent=6)
    out_f.close()


