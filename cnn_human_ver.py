import numpy
import cv2, os
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint,EarlyStopping
import tensorflow as tf
from tensorflow.python.keras.models import load_model

x_train = []
y_train = []
x_test = []
y_test = []

file_location = 'C:/Users/Kim Hyo Rim/Desktop/What Do I Look Like/CNN/'

# train data
animal = ["cat","dog","rabbit"]



idx = 0
for p in animal:

    print("> " + animal[idx] + " on Process")
    count = 0
    folder = file_location + 'train/' + animal[idx]

    for i in os.listdir(folder):
        if os.path.isfile(folder + "/" + i):

            try :    # 파일이름이 한글이면 error -> 영어로 rename
                new_name = folder + "/" + 'trainImage' + str(count) + '.jfif'
                os.rename(folder + "/" + i, new_name)
            except:
                pass

            img = cv2.imread(new_name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (50, 50))
            img = numpy.asarray(img) / 255
            x_train.append(img)
            y_train.append(idx)
            count = count + 1
        if count == 87:  # 200
            break
    idx += 1

idx = 0
for p in animal:

    print("> " + animal[idx]  + " on Process")
    count = 0
    folder = file_location + 'val/' + animal[idx]

    for i in os.listdir(folder):
        if os.path.isfile(folder + "/" + i):

            try:    # 파일이름이 한글이면 error -> 영어로 rename
                new_name = folder + "/" + 'testImage' + str(count) +  + '.jfif'
                os.rename(folder + "/" + i, new_name)
            except:
                pass

            img = cv2.imread(new_name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (50, 50))
            img = numpy.asarray(img) / 255
            x_test.append(img)
            y_test.append(idx)
            count = count + 1
        if count == 29:  # 30
            break
    idx += 1




x_train = numpy.array(x_train)
x_train = x_train.reshape(x_train.shape[0], 50, 50, 1)

x_test = numpy.array(x_test)
x_test = x_test.reshape(x_test.shape[0], 50, 50, 1)

y_train = numpy.asarray(y_train)
y_train = y_train.reshape(y_train.shape[0], 1)
y_train = np_utils.to_categorical(y_train)

y_test = numpy.asarray(y_test)
y_test = y_test.reshape(y_test.shape[0], 1)
y_test = np_utils.to_categorical(y_test)

# 3
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(50, 50, 1), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=8))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

MODEL_DIR = '../model/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

modelpath = "../model/{epoch:02d}-{val_loss:.4f}.h5"
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=40, batch_size=30, verbose=1, callbacks=[early_stopping_callback, checkpointer])

#test

test = [] # test input
cat2 = cv2.imread("C:/Users/Kim Hyo Rim/Desktop/sohee.jfif")
cat2 = cv2.cvtColor(cat2, cv2.COLOR_BGR2GRAY)
cat2 = cv2.resize(cat2, (50, 50))
cat2 = numpy.asarray(cat2) / 255
cat2 = numpy.array(cat2)
test.append(cat2)
test = numpy.array(test)
test = test.reshape((1, 50, 50, 1))

y_test = model.predict_classes(test)
print(y_test)