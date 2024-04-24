import numpy as np
from keras.models import Sequential
from keras.layers import Masking, Conv1D, GlobalMaxPooling1D, Dense, Dropout

# Function to load dataset
def load_dataset(data_path):
    return np.load(data_path)

# Function to create the CNN model
def create_model(input_shape, num_classes):
    model = Sequential([
        Masking(mask_value=0.0, input_shape=input_shape),
        Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        GlobalMaxPooling1D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

