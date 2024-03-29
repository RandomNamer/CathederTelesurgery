from utils.filesystem import recursive_mkdir
from configs import video_sample_config
import subprocess
from os.path import join, split


# output_dir = "/mnt/e/Workspace/CathederTelesurgery/Data/Datasets/RigidModelVideo-11-21/"
output_dir = "/mnt/e/Workspace/CathederTelesurgery/Data/Datasets/RigidModelVideo-2-20/"

for cfg in video_sample_config:
    src_path = cfg["path"]
    vid_file_name = split(src_path)[-1].split(".")[0]
    clip_counter = 0
    for clip in cfg['clips']:
        working_dir = join(output_dir, f'{vid_file_name}-clip{clip_counter}')
        cmd = f"ffmpeg -i {src_path} -ss {clip[0].to_seconds()} -to {clip[1].to_seconds()} -q:v 2 {working_dir}/frame%04d.jpg"
        recursive_mkdir(working_dir)
        print('Calling: ', cmd)
        res = subprocess.call(cmd, shell=True,)
        print(f'successfully processed clip {clip_counter} for video {vid_file_name} with {res}')
        clip_counter += 1
