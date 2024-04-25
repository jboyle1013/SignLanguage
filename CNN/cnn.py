import numpy as np
from keras.models import Sequential
from keras.layers import Masking, Conv1D, GlobalMaxPooling1D, Dense, Dropout, InputLayer
from glob import glob
from keras.callbacks import TensorBoard
import re
import os
from keras.utils import to_categorical
import collections


# to use TensorBoard to view training accuracy, cd to the ./Logs/train
# and run the command "tensorboard --logdir=."
# then copy and go to link that is given as the result
logDir = os.path.join("./Logs")
tbCallbacks = TensorBoard(log_dir=logDir)


# Function to load dataset
def load_dataset(data_path):
    return np.load(data_path)

class Model():

# Function to create the CNN model
    def create_model(self, sample_shape, num_classes):
        self.model = Sequential([
            InputLayer(sample_shape),
            #Masking(mask_value=0.0, input_shape=input_shape),
            Conv1D(64, kernel_size=3, activation='relu', padding='same',kernel_initializer='he_uniform'),
            Conv1D(64, kernel_size=3, activation='relu', padding='same',kernel_initializer='he_uniform'),
            GlobalMaxPooling1D(),
            Dense(128, activation='relu',kernel_initializer='he_uniform'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return self.model

    def fit_model(self, X_train, y_train):
        self.model.fit(X_train, y_train, epochs=200, callbacks=[tbCallbacks], batch_size=X_train.shape[0])
        
        model_list = glob("./model_*")
        if len(model_list) != 0:
            model_nums = [re.findall(r'\d+', file_name)[0] for file_name in model_list]
            new_model_num = max(map(int, model_nums)) + 1
        else:
            new_model_num = 0
        self.save_model(f"model_{new_model_num}.keras")
        
    # to print a breakdown of the layers, parameters and output shapes of each layer
    def summery(self):
        self.model.summary()
        
    
    def save_model(self, model_file_name):
        self.model.save(model_file_name)
    
    
    def load_model(self, model_file_name):
        self.model.load_weights(model_file_name)
    
    
    def predict(self, X_test):
        self.prob_predictions = self.model.predict(X_test)
        indexes_of_highest_prob_for_each_row = np.argmax(self.prob_predictions, axis=-1)
        
        # i believe the indexes correspond to the label numbers
        self.predictions = indexes_of_highest_prob_for_each_row
        return self.predictions


if '__main__' == __name__:
    input_train_dir = "./Data/data-subset/train/"
    # input_test_dir = "../Data/data-subset/train/"
    # input_val_dir = "../Data/data-subset/val/"
    y_train = load_dataset(input_train_dir + "labels.npy")
    features_npy_file_paths = glob(input_train_dir + "pFeatures/*")
    input_features = [load_dataset(npy) for npy in features_npy_file_paths]
    batch_size = 45
    X_train = np.array(input_features)
    print(X_train.shape)
    
    num_classes = len(collections.Counter(y_train.flatten()).items())
    # print(f"Num classes = {num_classes}\n")
    # print(y_train[:10])
    y_label_list = np.sort(np.unique(y_train))
    y_label_dic = {label:i for (label,i) in zip(y_label_list,range(len(y_label_list)))}
    y_train = np.array([y_label_dic[label] for label in y_train])
    y_train = to_categorical(y_train).astype(int)

    #print(y_train[:10])
    #print()
    #print(y_train.shape)
    model = Model()
    sample_shape = (X_train.shape[1],X_train.shape[2])
    model.create_model(sample_shape, num_classes)
    model.summery()
    train = []
    i = 0
    for _ in range(X_train.shape[0] // batch_size):
        train = X_train[i:i+batch_size]
        print(train.shape)
        model.fit_model(X_train, y_train)
        i += batch_size
    
    