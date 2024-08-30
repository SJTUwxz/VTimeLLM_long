import json
import os
from decord import VideoReader

if __name__ == "__main__":

    f = open("/mnt/mir/datasets/vlm-evaluation-datasets/Charades-STA/charades_sta_test.txt", "r")
    video_folder = "/mnt/mir/datasets/vlm-evaluation-datasets/Charades-STA/Charades_v1_480"
    save_f = open("/home/fengchan/projects-ng/xizi/VTimeLLM/data/vtimellm_eval/charades_sta_test.json", "w")

    info = {}

    for line in f.readlines():
        vid_start_end, sentence = line.split("##")
        vid, start, end = vid_start_end.split(' ')

        if vid not in info:
            vr = VideoReader(os.path.join(video_folder, str(vid)+'.mp4'))     
            fps_vid = vr.get_avg_fps()
            duration = float(len(vr)) / fps_vid
            info[vid] = {"duration": duration, "timestamps":[], "sentences":[]}

        info[vid]["timestamps"].append([float(start), float(end)])
        info[vid]["sentences"].append(sentence.strip())

    json.dump(info, save_f, indent=6)
    save_f.close()


            
