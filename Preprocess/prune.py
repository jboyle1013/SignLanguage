# This file is used to find the top 20 most frequent labels in the training set
# that are also in the testing and validation set. It prints out the frequencies
# and the labels and sequence IDs. It can also be used to copy all the sequencies
# from the original training, testing, and validation set, matching those top 20 labels
# into a newly created output directory. It shows progress bars for the copying done.
# This cleans and prunes the original data set.
# 
# Author: Ben Barber
# 
import os
from glob import glob
from shutil import copytree
from tqdm import tqdm
import mediapipe as mp
from mediapipe_preprocess_get_features import get_landmark_points, mediapipe_detection
import cv2

# Make mediapipe Model
mp_holistic = mp.solutions.holistic
mediapipe_model = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# returns a dictionary with keys as labels and values as lists of the sequence IDs
# that have that label 
def get_labels_and_sequence_IDs(input_dir):
    label_sequence_dic = {}
    for sequence_ID_filepath in glob(input_dir + "labels/bbox/*"):
        sequence_ID = os.path.basename(sequence_ID_filepath)
        first_bbox_filepath = glob(sequence_ID_filepath + "/image_*_bbox.txt")[0]
        with open(first_bbox_filepath) as first_image_bbox_file:
            first_line = first_image_bbox_file.readline()
            label = first_line.split(" ")[0]
            existing_list = label_sequence_dic.get(label)
            if existing_list:
                existing_list.append(sequence_ID)
            else:
                label_sequence_dic[label] = [sequence_ID]
    return label_sequence_dic



# gets top 20 labels and number of sequences having each of those labels as a list of tuples
# with label first and number of sequences second.
# It also prints these out nicely if print_data = True.
def get_top_20_labels_and_num_sequences(label_sequence_dic, print_data = False):
    sum = 0
    label_and_length_list = []
    for label_key, sequence_ID_list in label_sequence_dic.items():
        num_sequences_for_label = len(sequence_ID_list)
        label_and_length_list.append((label_key, num_sequences_for_label))
        sum += num_sequences_for_label

    label_and_length_list.sort(reverse=True,key=lambda List: List[1])

    if print_data:
        for i, (label, num_sequences) in enumerate(label_and_length_list):
            if i == 20:
                break
            print(f"{i}: Label = {label}, Number of sequences = {num_sequences}")


        print(f"\n\nSum of sequences = {sum}")
    return label_and_length_list[:20]

# This gets all of the labels and numbers of sequences having that label from the dictionary passed in that
# also matches one of the labels in the list passed in. This returns a dictionary of the matching labels as keys
# and the corresponding amount of sequences as the value. It also returns a list of the sequence IDs that were
# matched and prints out the data nicely.  
def get_labels_lengths_and_sequences_from_dict_matching_ones_in_list(top_20_label_and_length_list, label_sequence_dic, print_data = False):
    label_and_length_from_dict = {}
    matching_sequence_IDs = []
    for i,(label, _) in enumerate(top_20_label_and_length_list):
        sequences_for_label = label_sequence_dic[label]
        num_of_sequences_for_label_in_dict = len(sequences_for_label)
        label_and_length_from_dict[label] = (num_of_sequences_for_label_in_dict, sequences_for_label)
        matching_sequence_IDs.extend(sequences_for_label)
        if print_data:
            print(f"{i}: Label = {label}, Number of sequences = {num_of_sequences_for_label_in_dict} Sequecnes: {sequences_for_label}")
    return label_and_length_from_dict, matching_sequence_IDs

# this removes all the keys from remove_from_dict that are not also in based_on_dic
def remove_non_matching_keys(remove_from_dict, based_on_dic):
    keys = remove_from_dict.copy().keys()
    for key in keys:
        if key not in based_on_dic:
            del remove_from_dict[key]

# This copies the sequences and labels having the IDs in IDs_list from the input_dir to output_dir.
# This prints out a progress bar as well.  
def copy_frames_and_labels_matching_list_of_IDs(input_dir, output_dir, IDs_list):
    input_frames_dir = input_dir + "frames/"
    input_frames_paths = [input_frames_dir + id for id in IDs_list]
    output_frames_dir = output_dir + "frames/"
    max = 0
    for input_frames_path in tqdm(input_frames_paths, desc=f'Copying to "{output_dir}"'):
        sequence_ID = os.path.basename(input_frames_path)
        output_frames_path = output_frames_dir + f"{sequence_ID}"
        copytree(src=input_frames_path, dst=output_frames_path)
        labels_bbox_input_path = input_dir + f"labels/bbox/{sequence_ID}"
        labels_bbox_output_path = output_dir + f"labels/bbox/{sequence_ID}"
        copytree(src=labels_bbox_input_path,dst=labels_bbox_output_path)
        remove_extra_frames(output_frames_path)
        new = len(os.listdir(output_frames_path))
        if new > max:
            max = new
    print(f"\nmax frames = {max}\n")
    
    
def remove_extra_frames(frames_path):
    frame_paths = glob(frames_path + '/*')
    for frame_path in frame_paths:
        image = cv2.imread(frame_path)
        _, results = mediapipe_detection(image, mediapipe_model)
        features = get_landmark_points(results)
        if features[132:].any() == False:
            os.remove(frame_path)
            
        
    

if '__main__' == __name__:
    data_dir = "../datasolid20untrimmed/"
    input_dir_train = data_dir + "train/"
    input_dir_test = data_dir + "test/"
    input_dir_val = data_dir + "val/"
    
    train_dict = get_labels_and_sequence_IDs(input_dir_train)
    test_dict = get_labels_and_sequence_IDs(input_dir_test)
    val_dict = get_labels_and_sequence_IDs(input_dir_val)
    remove_non_matching_keys(train_dict, test_dict)
    remove_non_matching_keys(train_dict, val_dict)
    print("\n\n\nTop 20 most frequent labels from training set:\n\n")
    top_20_label_and_length_list = get_top_20_labels_and_num_sequences(train_dict, print_data=True)
    _, train_matching_top_20_sequence_IDs = get_labels_lengths_and_sequences_from_dict_matching_ones_in_list(top_20_label_and_length_list, train_dict)
    print("\n\n\nTop 20 most frequent labels from training set in testing set:\n\n")
    _, test_matching_top_20_sequence_IDs = get_labels_lengths_and_sequences_from_dict_matching_ones_in_list(top_20_label_and_length_list, test_dict, print_data=True)
    print("\n\n\nTop 20 most frequent labels from training set in validation set:\n\n")
    _, val_matching_top_20_sequence_IDs = get_labels_lengths_and_sequences_from_dict_matching_ones_in_list(top_20_label_and_length_list, val_dict, print_data=True)
    
    output_dir_train = "../set/train/"
    output_dir_test = "../set/test/"
    output_dir_val = "../set/val/"
    
    copy_frames_and_labels_matching_list_of_IDs(input_dir_train, output_dir_train, train_matching_top_20_sequence_IDs)
    copy_frames_and_labels_matching_list_of_IDs(input_dir_test, output_dir_test, test_matching_top_20_sequence_IDs)
    copy_frames_and_labels_matching_list_of_IDs(input_dir_val, output_dir_val, val_matching_top_20_sequence_IDs)
