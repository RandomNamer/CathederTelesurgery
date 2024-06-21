from utils.filesystem import recursive_mkdir
from configs import video_sample_config, output_dir
import subprocess
from os.path import join, split




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
        if len(clip) > 2:
            if clip[2] is not None:
                print(f'Clip {clip_counter} has desc: {clip[2]}')
                with open(f'{working_dir}/desc.txt', 'w') as f:
                    f.write(clip[2])
        print(f'successfully processed clip {clip_counter} for video {vid_file_name} with {res}')
        clip_counter += 1
