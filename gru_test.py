from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM, GRU, Dropout, Dense
from keras.models import Sequential

import load_data as ld
import numpy as np
from sklearn import preprocessing
from keras.utils import np_utils

# x_train, y_train, x_test = ld.load_data('Data')

# x_train.dump("x_train")
# y_train.dump("y_train")
# x_test.dump("x_test")

x_train = np.load("x_train")
y_train = np.load("y_train")
x_test = np.load("x_test")
all_data = np.concatenate((x_train, x_test))

###
# apply some preprocessing scaling
min_max_scaler = preprocessing.MinMaxScaler()
# X_train_minmax = min_max_scaler.fit_transform(all_data)
# x_train = X_train_minmax[:15485, :]
# x_test = X_train_minmax[:3871, :]
###
# OneHotEncoding
y_train_hot = np_utils.to_categorical(y_train)
###

print(y_train_hot)

print(x_train.shape, y_train.shape, x_test.shape, y_train_hot.shape)

model = Sequential()
model.add(GRU(360, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(GRU(360, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(180, activation='relu'))
model.add(Dense(y_train_hot.shape[1], activation='softmax'))
model.load_weights("./weights-improvement-09-0.3494")
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# define the checkpoint
filepath = "./weights-improvement-{epoch:02d}-{loss:.4f}"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# predictions = model.predict_proba(x_test)
import pickle

# file_output = open("prob_pred", "wb")
# pickle.dump(predictions, file_output)
# print(predictions)

input_file = open("prob_pred", "rb")
predictions_loaded = pickle.load(input_file)

output = []

for i, vector in enumerate(x_test):
    temp = np.hstack((vector[0][0], predictions_loaded[i][1]))
    output.append(temp)

print(output)

file_submission = open("output.csv", "w")
for i, vector in enumerate(output):
    file_submission.write(str(int(i+1)) + "," + str(vector[1]) + "\n")
#
# model.fit(x_train, y_train_hot, nb_epoch=10, batch_size=16, callbacks=callbacks_list, validation_split=0.1)
