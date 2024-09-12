import os
root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
import sys
sys.path.append(root_dir)
import json
from decord import VideoReader
from tqdm import tqdm

if __name__ == "__main__":

    video_folder = "/mnt/mir/datasets/vlm-datasets/momentor-10M/videos"
    annotation = json.load(open('/mnt/mir/datasets/vlm-datasets/momentor-10M/annotations/GESM_data.format.json','r'))

    medium_length = 20
    num_input_words = 0
    num_output_words = 0

    num_captions_segment = {10: 5, 20: 4, 30: 2, 40: 2, 50: 1, 60: 1, 70: 1, 80: 1, 90: 1, 100: 1}

    for vid in tqdm(list(annotation.keys())):

        try:
            vr = VideoReader(os.path.join(video_folder, str(vid)+'.mp4'))     
            fps_vid = vr.get_avg_fps()
            duration = float(len(vr)) / fps_vid
        except:
            continue

        captions = annotation[vid]['captions']
        timestamps = annotation[vid]['timestamps']

        num_captions = len(captions)
        total_word_count = 0
        for i in range(num_captions):
            total_word_count += len(captions[i].split(' '))

        if num_captions == 0:
            print(vid, annotation[vid])
            continue

        avg_word_count = total_word_count / num_captions
        avg_duration = float(100) / num_captions

        for percent in range(10, 100, 10):

            # percent is the length of segment to be generated
            num_sentences_needed = int(percent / avg_duration)
            if num_sentences_needed <= 1:
                continue

            num_input_words += num_sentences_needed * avg_word_count * num_captions_segment[percent]
            num_output_words += 20 * num_captions_segment[percent]


    print(int(num_input_words), int(num_output_words))
            

            


        

