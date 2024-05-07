import numpy as np
from keras.models import Sequential
from keras.layers import Masking, Conv1D, GlobalMaxPooling1D, Dense, Dropout, InputLayer, BatchNormalization
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
            Conv1D(64, kernel_size=3, activation='relu', padding='same',kernel_initializer='he_uniform'),
            Conv1D(64, kernel_size=3, activation='relu', padding='same',kernel_initializer='he_uniform'),
            GlobalMaxPooling1D(),
            Dense(128, activation='relu',kernel_initializer='he_uniform'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return self.model

    def fit_model(self, X_train, y_train, validation = None):
        if validation != None:
            self.model.fit(X_train, y_train, epochs=400, callbacks=[tbCallbacks], batch_size=X_train.shape[0], validation_data=validation, validation_batch_size=validation[0].shape[0])
        else:
            self.model.fit(X_train, y_train, epochs=400, callbacks=[tbCallbacks], batch_size=X_train.shape[0])
            
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

# this function loads the labels and features from the correct npy files given
# the input directory path to ex. train/ or test/ or val/ and returns the 
# corresponding X matrix and y matrix
# 
def get_X_and_y_from_dir(input_dir):
    y = load_dataset(input_dir + "labels.npy")
    features_npy_file_paths = glob(input_dir + "pFeatures/*")
    input_features = [load_dataset(npy) for npy in features_npy_file_paths]
    X = np.array(input_features)
    return X, y

# This function returns a one hot encoded y matrix and a list of the labels
# where the index correspondes to the one hot encoded placement.
# If you want a label for the one hot encoded 1000 then use
# y_label_list[np.argmax([1,0,0,0])] or y_label_list[0] to get the label
# since 0 corresponds to the index of the 1 in the one hot code.
# The y_label_dic is also returned to get the index given the label as a key
def get_one_hot_encoded_y_and_corresponding_list(y, num_classes = None):
    y_label_list = np.sort(np.unique(y))
    y_label_dic = {label:i for (label,i) in zip(y_label_list,range(len(y_label_list)))}
    y = np.array([y_label_dic[label] for label in y])
    if num_classes == None:
        y_one_hot = to_categorical(y).astype(int)
    else:
        y_one_hot = to_categorical(y, num_classes).astype(int)
    return y_one_hot, y_label_list, y_label_dic

if '__main__' == __name__:
    data_dir = "../set/"
    input_train_dir = data_dir + "train/"
    input_test_dir = data_dir + "test/"
    input_val_dir = data_dir + "val/"
    
    # get training data
    X_train, y_train = get_X_and_y_from_dir(input_train_dir)
    
    # number of classes
    num_classes = len(collections.Counter(y_train.flatten()).items())
    
    # get testing and validation data
    X_test, y_test = get_X_and_y_from_dir(input_test_dir)
    X_val, y_val = get_X_and_y_from_dir(input_val_dir)
    
    # turn y traing into one hot encoded
    y_one_hot_train, y_train_label_one_hot_list, y_train_label_one_hot_dict = get_one_hot_encoded_y_and_corresponding_list(y_train)
    
    # correspond y validation values with new labels from y_train so they can be one hot encoded 
    y_val_corresponding = np.array([y_train_label_one_hot_dict[label] for label in y_val.flatten()])
    
    # finish one hot encoding y validation 
    y_one_hot_val = to_categorical(y_val_corresponding, num_classes=num_classes).astype(int)

    print(f"Num classes = {num_classes}\n") 
    print(f"X_train {X_train.shape}")
    print(f"X_test {X_test.shape}")
    print(f"X_val {X_val.shape}")
    
    print(f"y_train {y_train.shape}")
    print(f"y_test {y_test.shape}")
    print(f"y_val {y_val.shape}")
    
    # create and train model with validation at each epoch
    model = Model()
    sample_shape = (X_train.shape[1],X_train.shape[2])
    model.create_model(sample_shape, num_classes)
    model.summery()
    validation = (X_val,y_one_hot_val)
    model.fit_model(X_train, y_one_hot_train, validation)
    
    
    
    
    
    
    # y_train = load_dataset(input_train_dir + "labels.npy")
    # features_npy_file_paths = glob(input_train_dir + "pFeatures/*")
    # input_features = [load_dataset(npy) for npy in features_npy_file_paths]
    # X_train = np.array(input_features)
    # print(f"X_train {X_train.shape}")
    
    # num_classes = len(collections.Counter(y_train.flatten()).items())
    # print(f"Num classes = {num_classes}\n")
    # print(f"y_train {y_train.shape}")
    # # print(y_train[:10])
    # y_label_list = np.sort(np.unique(y_train))
    # y_label_dic = {label:i for (label,i) in zip(y_label_list,range(len(y_label_list)))}
    # y_train = np.array([y_label_dic[label] for label in y_train])
    # y_train = to_categorical(y_train).astype(int)
    # print(f"y_train {y_train.shape}")

    # y_test = load_dataset(input_test_dir + "labels.npy")
    # features_npy_file_paths = glob(input_test_dir + "pFeatures/*")
    # input_features = [load_dataset(npy) for npy in features_npy_file_paths]
    # X_test = np.array(input_features)
    
    
    # y_label_list = np.sort(np.unique(y_test))
    # y_label_dic = {label:i for (label,i) in zip(y_label_list,range(len(y_label_list)))}
    # y_test = np.array([y_label_dic[label] for label in y_test])
    # y_test = to_categorical(y_test).astype(int)
    # print(f"y_test {y_test.shape}")
    #print(y_train[:10])
    #print()
    #print(y_train.shape)
    # model = Model()
    # sample_shape = (X_train.shape[1],X_train.shape[2])
    #model.create_model(sample_shape, num_classes)
    #model.summery()
    #validation = (X_val,y_val)
    #model.fit_model(X_train, y_train, validation)
    # train = []
    # i = 0
    # for _ in range(X_train.shape[0] // batch_size):
    #     train = X_train[i:i+batch_size]
    #     print(f"train {train.shape}")
    #     validation = (X_test[i:i+batch_size],y_test[i:i+batch_size])
    #     #model.fit_model(X_train, y_train, validation)
    #     i += batch_size
    
    