import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model




def load_dataset():
    X = []
    y = []
    root_dir = "./dataset/train"
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(subdir, file)
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (64, 64))
            X.append(img)
            class_folder = os.path.basename(subdir)
            if class_folder == "Open":
                y.append(0)
            elif class_folder == "Closed":
                y.append(1)
    return np.array(X), np.array(y)

X, y = load_dataset()

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("xtrain\n",X_train,"len :",len(X_train))
print("ytrain\n",y_train,"len :",len(y_train))
print("xtest\n",X_test,"len :",len(X_test))
print("ytest\n",y_test,"len :",len(y_test))

# One-hot encoding the target variable
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Reshaping the input data
X_train = X_train.reshape(X_train.shape[0], 64, 64, 1)
X_test = X_test.reshape(X_test.shape[0], 64, 64, 1)

# Defining the model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

# Compiling the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fitting the model
model.fit(X_train, y_train, steps_per_epoch = 5,
                      epochs = 30, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Accuracy:", accuracy)



model.save("dnn_model.h5",overwrite=True)


