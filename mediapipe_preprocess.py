# file: mediapipe_preprocessing.py
# author: Ben Barber
# description: This file goes through all the sequences of frames
#              and the corresponding label file, it uses mediapipe
#              to get feature points of hands, face, and pose and 
#              then makes an input matrix of size 
#              (num_sequences, num_frames_per_sequence, num_features_per_frame)
#              and a hot ones encoding of the corresponding label array.
#              This is intended to be used for training the model.

import cv2
import os
from keras.utils import to_categorical
import numpy as np
import mediapipe as mp


# Gets features (landmark points for coordinates (x,y,z) of pose, face, left hand, right hand)
# If landmark was not registered/seen then a flattened array of zeros matching its length is used 
def get_landmark_points(results):
    num_points_per_landmark = 3
    num_points_per_pose_landmark = 4
    num_landmarks_per_hand = 21
    num_landmarks_per_pose = 33
    num_landmarks_per_face = 468
    

    # get pose features
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() \
    if results.pose_landmarks else np.zeros(num_landmarks_per_pose*num_points_per_pose_landmark)
    
    # get face features
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() \
    if results.face_landmarks else np.zeros(num_landmarks_per_face*num_points_per_landmark)
    
    # get left hand features
    left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() \
    if results.left_hand_landmarks else np.zeros(num_landmarks_per_hand*num_points_per_landmark)
    
    # get right hand features
    right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
    if results.right_hand_landmarks else np.zeros(num_landmarks_per_hand*num_points_per_landmark)
    
    # return flattened features concatenated together
    return np.concatenate([pose, face, left_hand, right_hand])

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image.flags.writeable = False                  
    results = model.process(image)                 # Make landmark prediction
    image.flags.writeable = True                   
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    return image, results

mp_holistic = mp.solutions.holistic


#  this will be used to store/save the training and
#  label data numpy arrays X and y
output_dir = "./processed/"  # path to output directory

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# these should be changed from test to training when actually training
folder = "test" # or test
images_path = f"./data/MS-ASL/frames/{folder}/images/"
labels_path = f"./data/MS-ASL/frames/{folder}/labels/"
sequences = []
labels = []
count = 0
skips = 0


sequence_file_names = os.listdir(images_path)
for sequence_file_name in sequence_file_names:
    count += 1
    features_sets = []
    sequence_file_path = os.path.join(images_path, sequence_file_name)
    frame_file_names =  os.listdir(os.path.abspath(sequence_file_path))
    if (len(frame_file_names) != 30):
        skips += 1
        print(f"SKIPPING {count}: {sequence_file_name}")
        print(f"Must be 30 frames in sequence but there were {len(frame_file_names)}\n\n")
    else:
        for frame_file_name in frame_file_names:
            frame_file_path = os.path.join(sequence_file_path, frame_file_name)
            image = cv2.imread(frame_file_path)
            
            # Get feature set (landmark points)
            try:
                _, results = mediapipe_detection(image, holistic)
            except Exception as e:
                print(f"FAIL: {sequence_file_name}         {frame_file_name}\n\n")
                print(e)
            features = get_landmark_points(results)
            
            # # DEBUG: sanity check for feature length
            # if (len(features) != 1662):
            #     print(f"FAIL on feat. length {len(features)}: {sequence_file_name}         {frame_file_name}\n\n")
            #     exit()
                
            features_sets.append(features)
            
        sequences.append(features_sets)
        
        label_file_name = f"{labels_path}{os.path.basename(sequence_file_name)}.txt"
        
        with open(label_file_name,'r') as label_file:
            # get first line in file
            line = label_file.readline() 
            # get the first number (label) from line
            label = line.split(" ",1)[0]
            labels.append(label)


# input for training and hot ones encoding of coresponding labels

# input matrix (num_sequences, num_frames_per_sequence, num_features_per_frame)
X = np.array(sequences)

# hot ones encoding of label array
y = to_categorical(np.array(labels)).astype(int)

print(f"X.shape = {X.shape}")
print(f"y.shape = {y.shape}")
print(f"sequences checked = {count}")
print(f"sequences skipped = {skips}")
print(f"sequences kept = {X.shape[0]}")

# save npz file of X matrix and y matrix
# these are stored as np.arrays and can be easily accessed through the names 'X' and
# 'y' after loading, np.load(), the npz file
np.savez_compressed("X_and_y", X = X, y = y)


   
   