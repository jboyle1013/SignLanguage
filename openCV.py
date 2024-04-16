# file: openCV.py
# author: Ben Barber
# description: This file gets live video from the computer camera
#              and displays it as well as extracts frames from the 
#              video intented for processing and prediction

import cv2
import mediapipe as mp
import numpy as np
from sequential_lstm import define_model


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
prediction_word = "Hello"
font_coord = (3, 30)
black_bgr = (0, 0, 0)
esc = 27




def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image.flags.writeable = False                  
    results = model.process(image)                 # Make landmark prediction
    image.flags.writeable = True                   
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    return image, results


def draw_landmarks(image, results):

    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             )

    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(0,153,0), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2)
                             ) 

    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
                             ) 

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





cv2.namedWindow("preview", cv2.WND_PROP_FULLSCREEN)
sequence = []
vc = cv2.VideoCapture(0)
if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
else:
    rval = False
    print("Error opening video stream or file")
    

while rval:
    rval, frame = vc.read()
    if rval:
        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        
        # Draw landmarks
        draw_landmarks(image, results)
        
        # Get feature set (landmark points)
        landmark_points = get_landmark_points(results)
        
        # append feature set of frame to make sequence 
        sequence.append(landmark_points)
        
        # only keep feature sets (sets of landmark points) for last 30 frames
        sequence = sequence[-30:]
        
        # # only predict once data has been collected for 30 frames (full sequence)
        # if len(sequence) == 30:
        #     prediction_probabilities = model.predict(np.expand_dims(sequence, axis=0))[0]
        #     prediction_word = categories[np.argmax(prediction_probabilities)]
        
        # flip for selfie view
        image = cv2.flip(image, 1)
        
        # put prediction word on image
        cv2.putText(image, prediction_word, font_coord, 
                cv2.FONT_HERSHEY_SIMPLEX, 1, black_bgr, 2, cv2.LINE_AA)
        
        # Display the resulting image
        cv2.imshow("preview", image)
        key = cv2.waitKey(20)
        
        
        # exit on esc or close button
        if key == esc:
            esc = True
            break
        try:
            # throws exception if window was closed
            cv2.getWindowProperty('preview', 0)
        except Exception:
            break # if window was closed

if esc is True:   
    cv2.destroyWindow("preview")
vc.release()