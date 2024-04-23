from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, LSTM

model = Sequential()
# A -> Number of frames (Temporal Dimension)
# B -> Number of mediapipe detected.
# 3 -> x, y, z of the points that medipipe detected
model.add(Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu', input_shape=(A, B, 3)))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))

model.add(Conv3D(filters=128, kernel_size=(3, 3, 3), activation='relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))

model.add(Conv3D(filters=256, kernel_size=(3, 3, 3), activation='relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))

# To connect a 3D layer to a dense layer, need to flatten the 3D output to 1D
model.add(Flatten())

# Adding recurrent layers is optional and can be considered if we flatten the 3D convolutions
# model.add(LSTM(units=128, return_sequences=True))
# model.add(LSTM(units=64))

model.add(Dense(units=1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=num_classes, activation='softmax'))  # num_classes -> number of gestures

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
