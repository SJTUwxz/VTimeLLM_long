import subprocess
import os
from pathlib import Path
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import csv
from ipdb import set_trace
import pickle as pkl
logging.basicConfig(filename="std.log", 
					format='%(asctime)s %(message)s', 
					filemode='w') 

logger = logging.getLogger()


def check_if_exist(dir, all_yt_name):

	non_exist = []
	for tmp_name in all_yt_name:
		file_path = dir + '/{}.mp4'.format(tmp_name)
		if not os.path.exists(file_path):
			non_exist.append(tmp_name)

	return non_exist

	

def download_clip(
	video_identifier,
	output_filename,
	out_compress_path,
	num_attempts=1,
	url_base='https://www.youtube.com/watch?v='
):
	status = False

	command = f"""
	yt-dlp --sleep-interval 1 --user-agent "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36" --username oauth2 --password '' --quiet --no-overwrites --no-warnings -S "vcodec:h264/acodec:m4a" -f "bestvideo[height>=360][height<=480]+bestaudio/best[height<=720]/best" --recode-video mp4 --force-keyframes-at-cuts --postprocessor-args "-c:a aac -b:a 128k -ar 16000" -o "{output_filename}" "{url_base}{video_identifier}"
	""".strip()

	attempts = 0
	while True:
		time.sleep(10)
		try:
			output = subprocess.check_output(command, shell=True,
											 stderr=subprocess.STDOUT)
			print(f"Command output for {video_identifier}:")
			print(output.decode('utf-8'))
		except subprocess.CalledProcessError as err:
			attempts += 1
			print(f"Error for {video_identifier}: {err.output.decode('utf-8')}")
			if attempts == num_attempts:
				return status, err.output
		else:
			break

	# Check if the video was successfully saved.
	status = os.path.exists(output_filename)

	if not status:
		print("Fail:", command) 
	return status, 'Downloaded'


def main(
	data_dir: str,
	sampling_rate: int = 44100,
	yt_list=None
):
	"""
	Download the clips within the MusicCaps dataset from YouTube.

	Args:
		data_dir: Directory to save the clips to.
		sampling_rate: Sampling rate of the audio clips.
	"""

	data_dir = Path(data_dir)
	data_dir.mkdir(exist_ok=True, parents=True)

	def process(yt_info):
		outfile_path = str(data_dir / f"{yt_info}.mp4")
		outfile_path2 = str(data_dir / f"{yt_info}.mp4")
		status = True
		if not os.path.exists(outfile_path):
			status = False
			status, log = download_clip(
				yt_info,
				outfile_path,
				outfile_path2
			)
		return yt_info
	
	with tqdm(total=len(yt_list)) as pbar:
		with ThreadPoolExecutor(max_workers=8) as ex:
			futures = [ex.submit(process, url) for url in yt_list]
			for future in as_completed(futures):
				result = future.result()
				pbar.update(1)


if __name__ == '__main__':
	all_yt_name = []
	
	count_line = 0
    video_names = pkl.load(open("./download_ids.pkl", "rb"))
    exist_ids = pkl.load(open("./exist_ids.pkl", "rb"))
    to_download = [x for x in video_names if x not in exist_ids]
    for row in to_download:
        all_yt_name.append(f'https://www.youtube.com/watch?v={row}')
	
	data_dir = '~/satoshithesis/InternVid-10M-YTB/'



	new_all_yt_name = check_if_exist(data_dir, all_yt_name)
	
	

	# Save the list to a CSV file
	# with open('res_30k.csv', mode='w', newline='') as file:
	# 	writer = csv.writer(file)
	# 	for item in new_all_yt_name:
	# 		writer.writerow([item])
	# set_trace()
	main(
		data_dir,
		sampling_rate=16000,
		yt_list=new_all_yt_name
	)


#yt-dlp --user-agent "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36" --username oauth2 --password '' -S "vcodec:h264/acodec:m4a"  -f "bestvideo[height<=240]+bestaudio/best[height<=360]/best[height<=480]/best[height<=720]"  --recode-video mp4 --download-sections "*10-20" --force-keyframes-at-cuts  --postprocessor-args "-c:a aac -b:a 128k -ar 16000" -o "ggg.mp4"  "https://www.youtube.com/watch?v=-ZJqu_4zLMc"


