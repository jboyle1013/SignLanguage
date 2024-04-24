import json
import cv2
import os
import pandas as pd
import shutil
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import string


def load_word_frequencies(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data

top_words = load_word_frequencies('top_100word_frequencies.json')


def is_word_in_list(input_string, word_list):
    # Remove punctuation and convert to lower case
    translator = str.maketrans('', '', string.punctuation)
    cleaned_string = input_string.translate(translator).lower()
    words = cleaned_string.split()
    # Convert word list to lower case
    word_list = [word.lower() for word in word_list]
    # Check if any word from the input string is in the word list
    return any(word in word_list for word in words)

def extract_frames_from_video(video_path, output_folder, sentence_id):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    success, image = cap.read()
    frame_count = 0

    while success:
        frame_path = os.path.join(output_folder, f"{sentence_id}_{frame_count:04d}.jpg")
        cv2.imwrite(frame_path, image)
        success, image = cap.read()
        frame_count += 1

    cap.release()
    return frame_count

def fix_times(start_time, end_time):
    new_end = end_time - start_time
    new_start = 0
    return new_start, new_end

def estimate_word_frames(start_time, end_time, fps, total_frames):
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    end_frame = min(end_frame, total_frames - 1)
    return range(start_frame, end_frame + 1)

def move_and_rename_frames(frame_range, orig_folder, output_folder, sentence_id, s_name, word_id):
    new_dir_path = os.path.join(output_folder, f"{sentence_id}", f"{word_id}")
    if not os.path.exists(new_dir_path):
        os.makedirs(new_dir_path)

    for i, frame_number in enumerate(frame_range):
        old_frame_path = os.path.join(orig_folder, f"{s_name}_{frame_number:04d}.jpg")
        new_frame_path = os.path.join(new_dir_path, f"{sentence_id}.{word_id}.{s_name}_{frame_number:04d}.jpg")
        if os.path.exists(old_frame_path):
            shutil.copy(old_frame_path, new_frame_path)
        else:
            print(f"Missing frame: {old_frame_path}")

def encode_word(input_string):
    byte_array = input_string.encode('utf-8')
    int_representation = int.from_bytes(byte_array, 'big')
    chars = '0123456789abcdefghijklmnopqrstuvwxyz'
    result = []
    while int_representation > 0:
        int_representation, remainder = divmod(int_representation, 36)
        result.append(chars[remainder])
    return ''.join(reversed(result))

def process_folder(folder):
    print(f'Working with folder: {folder}')
    fps = 30
    base_video_path = f'h2signdata/How2Sign/sentence_level/{folder}/raw_videos'
    base_output_path = f'h2signdata/How2Sign/sentence_level/{folder}/frames'
    csv_path = f'h2signdata/How2Sign/sentence_level/{folder}/text/how2sign_{folder}.csv'

    annotations = pd.read_csv(csv_path, sep='\t')

    for index, row in tqdm(annotations.iterrows(), total=len(annotations), desc=f"Processing rows in {folder}"):
        sentence = row['SENTENCE']
        # res = is_word_in_list(sentence, top_words.keys())
        # if res:
        frames_path = f'h2signdata/How2Sign/frames/{folder}/frames'
        sentence_id = row['SENTENCE_NAME']
        video_path = os.path.join(base_video_path, f"{row['SENTENCE_NAME']}.mp4")
        output_folder = os.path.join(frames_path, f"{row['SENTENCE_ID']}")

        total_frames = extract_frames_from_video(video_path, output_folder, sentence_id)

        new_start, new_end = fix_times(row['START'], row['END'])
        words = row['SENTENCE'].split()
        time_per_word = (new_end - new_start) / len(words)

        for word_index, word in enumerate(words):
            start_time = new_start + word_index * time_per_word
            end_time = start_time + time_per_word
            fixed_start, fixed_end = fix_times(start_time, end_time)
            word_frame_range = estimate_word_frames(fixed_start, fixed_end, fps, total_frames)
            word_id = encode_word(word)
            move_and_rename_frames(word_frame_range, output_folder, output_folder, row['SENTENCE_ID'], sentence_id, word_id)

if __name__ == "__main__":
    # folders = ['train', 'test', 'val']
    folders = ['train']
    with ProcessPoolExecutor() as executor:
        executor.map(process_folder, folders)
