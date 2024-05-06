from video_downloader import *
from preprocess import *

def main():
    WSL_PATH = "/mnt/c/Users/Jordan/Documents/Machine Learning/WLASL-master"
    # download_nonyt_videos('WLASL-master/start_kit/splits/asl300.json', 'WLASL-master/WLASL/data/raw_videos')
    # check_youtube_dl_version()
    # download_yt_videos(f'{WSL_PATH}/WLASL-master/start_kit/splits/asl300.json', f'{WSL_PATH}/WLASL-master/WLASL/data/raw_videos_yt')

    # convert_everything_to_mp4()

    content = json.load(open('WLASL-master/start_kit/splits/asl300.json'))
    extract_all_yt_instances(content)

    # delete_empty_subfolders()



if __name__ == '__main__':
    main()