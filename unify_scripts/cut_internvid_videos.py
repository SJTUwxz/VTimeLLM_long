from moviepy.video.io.VideoFileClip import VideoFileClip
import os
import json
from tqdm import tqdm
from multiprocessing import Pool
import time
import ffmpeg

# def cut_video(input_file, output_file, start_time, end_time):
def cut_video(data):

    id = data["id"]
    start_second, end_second = data["meta"]["split"]

    input_path = './InternVid-10M-YTB'
    output_path = './InternVid-cut-videos'
    video_path = os.path.join(input_path, id+".mp4")
    out_path = os.path.join(output_path, id+".mp4")

    if os.path.exists(out_path):
        return
    else:
        open(out_path, 'w').close()

    if os.path.exists(video_path):
        video = VideoFileClip(video_path)
        
        video_cut = video.subclip(start_second, end_second)
        
        video_cut.write_videofile(out_path, codec="libx264", audio_codec="aac")
        
        # Close the video to release resources
        video_cut.close()


def seconds_to_hhmmss(seconds):
    return time.strftime('%H:%M:%S', time.gmtime(seconds))

def cut_video_ffmpeg(data):

    id = data["id"]
    start_second, end_second = data["meta"]["split"]

    input_path = './InternVid-10M-YTB'
    output_path = './InternVid-cut-videos-ffmpeg'
    video_path = os.path.join(input_path, id+".mp4")
    out_path = os.path.join(output_path, id+".mp4")

    if os.path.exists(video_path):
        start_time = seconds_to_hhmmss(start_second)
        end_time = seconds_to_hhmmss(end_second)
        (
        ffmpeg
        .input(video_path, ss=start_time, to=end_time)
        .output(out_path, c='copy')
        .run()
    )


js = json.load(open('stage2.json'))

pool = Pool(16)
pool.map(cut_video_ffmpeg, js)
# for video_name in video_names:
print("Finished.")



