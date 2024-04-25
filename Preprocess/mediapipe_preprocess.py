import os
from PIL import Image
import mediapipe as mp
import numpy as np
import cv2

'''
This script is to be used on a preprocessed directory. It will use mediapipe to extract the features.
'''

features_directory = '../Data/data-subset/train/pFeatures'

'''
This will create a tensor of shape [3, num_landmarks, num_features]
where the first dimension is for pose, left hand, and riht hand
''' 
def get_landmark_points(results):
    num_points_per_landmark = 3
    num_points_per_pose_landmark = 4
    num_landmarks_per_hand = 21
    num_landmarks_per_pose = 33    

    # get pose features
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() \
    if results.pose_landmarks else np.zeros(num_landmarks_per_pose*num_points_per_pose_landmark)

    
    # get left hand features
    left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() \
    if results.left_hand_landmarks else np.zeros(num_landmarks_per_hand*num_points_per_landmark)
    
    # get right hand features
    right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
    if results.right_hand_landmarks else np.zeros(num_landmarks_per_hand*num_points_per_landmark)
    
    # return flattened features concatenated together
    return np.concatenate([pose, left_hand, right_hand])

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image.flags.writeable = False                  
    results = model.process(image)                 # Make landmark prediction
    image.flags.writeable = True                   
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    return image, results

# Go through all the frames in a folder and make extract feautres and pad.
def process_video_and_pad_features(video_folder, model, target_length=201):
    features_list = []
    # Loop through each frame file in the sorted order
    frame_files = sorted([f for f in os.listdir(video_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    for frame_file in frame_files:
        img_path = os.path.join(video_folder, frame_file)
        image = cv2.imread(img_path)
        _, results = mediapipe_detection(image, model)
        features = get_landmark_points(results)
        print(features.shape)
        exit()
        features_list.append(features)
    
    # Convert list of features to a numpy array
    # Pad the features array to ensure all are the same length
    if len(features_list) < target_length:
        # Calculate the number of features in each frame
        feature_length = len(features_list[0]) if features_list else 0  # Handle case with no frames
        padding = np.zeros((target_length - len(features_list), feature_length))
        features_array = np.vstack((features_list, padding))
    else:
        features_array = np.array(features_list[:target_length])

    return features_array

def process_all_videos(root_folder, output_folder, model):
    # Ensure the output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Process each subdirectory within the root folder
    video_folders = [os.path.join(root_folder, d) for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d))]
    for video_folder in video_folders:
        video_num = os.path.basename(video_folder)
        video_features = process_video_and_pad_features(video_folder, model)
        # Save the features to a numpy file
        np.save(os.path.join(output_folder, f"{video_num}.npy"), video_features)

# Make mediapipe Model
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

image_folder = './Data/data-subset/train/processed'
output_dir = './Data/data-subset/train/pFeatures'
os.makedirs(output_dir, exist_ok=True)
process_all_videos(image_folder, output_dir, holistic)