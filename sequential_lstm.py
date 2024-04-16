# file: sequenctial_lstm.py
# author: Ben Barber
# description: This file defines a sequenctial model built from LSTM and dense layers.


import os
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard

# to use TensorBoard to view training accuracy, cd to the ./Logs/train
# and run the command "tensorboard --logdir=."
# then copy and go to link that is given as the result
logDir = os.path.join("./Logs")
tbCallbacks = TensorBoard(log_dir=logDir)


class Model:
    # define and compile Sequential model of layers of LSTM and Dense with multi classification loss function
    # categorical_crossentropy
    # Note: Check that num_catagories is actually 101, that was just gotten from the yaml file
    def define_model(self, num_catagories = 101, num_frames_per_vid = 30, num_features_per_frame = 1662):
        self.model = Sequential()
        self.model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(num_frames_per_vid,num_features_per_frame)))
        self.model.add(LSTM(128, return_sequences=True, activation='relu'))
        self.model.add(LSTM(64, return_sequences=False, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(num_catagories, activation='softmax'))
        self.model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        
    
    # fit model and log TensorBoard callbacks
    # 154 epochs might overtrain or undertrain the model so use
    # TensorBoard callback logs to see accuracy vs epoch number
    # and then change this 
    def fit_model(self, X_train, y_train):
        self.model.fit(X_train, y_train, epochs=1, callbacks=[tbCallbacks])
        
    # to print a breakdown of the layers, parameters and output shapes of each layer
    def summery(self):
        self.model.summary()
        
    
    def save_model(self, model_file_name):
        self.model.save(model_file_name)
    
    def load_model(self, model_file_name):
        self.model.load_weights(model_file_name)
    
    # returns 
    def predict(self, X_test):
        self.prob_predictions = self.model.predict(X_test)
        indexes_of_highest_prob_for_each_row = np.argmax(self.prob_predictions, axis=-1)
        
        # i believe the indexes correspond to the label numbers
        self.predictions = indexes_of_highest_prob_for_each_row
        return self.predictions
    
    # Specify y_predictions if not using the predictions the model just
    # predicted, ie. self.predictions is used if y_predictions not given.
    # If y_test is one hot encoded then set from_one_hot = True,
    # otherwise, if it is the label nums, leave it as False.
    # @return accuracy as a decimal representing precentage ex. 0.0 to 1.0
    def accuracy(self, y_test, from_one_hot = False, y_predictions = None) -> float:
        if y_predictions == None:
            y_predictions = self.predictions
        if from_one_hot == True:
            y_test = from_categorical(y_test)
            
        correct_matrix = np.array(y_predictions) == np.array(y_test)

        num_correct = np.count_nonzero(correct_matrix.astype(int))

        accuracy = float(num_correct) / len(y_test)
        return accuracy
        
        
        
# returns original y array by converting the one hot encoding
# back to original labels by using argmax to return the index
# of the 1 in each row, which is the original label number
def from_categorical(one_hot_encoding_y):
    return np.argmax(one_hot_encoding_y, axis=-1)