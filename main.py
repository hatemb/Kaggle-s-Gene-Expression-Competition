from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM, GRU, Dropout, Dense
from keras.models import Sequential

import load_data as ld
import numpy as np
from sklearn import preprocessing
from keras.utils import np_utils

x_train, y_train, x_test = ld.load_data('Data')

# OneHotEncoding
y_train_hot = np_utils.to_categorical(y_train)
#
print(x_train.shape, y_train.shape, x_test.shape, y_train_hot.shape)

model = Sequential()
model.add(GRU(360, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(GRU(360, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(180, activation='relu'))
model.add(Dense(y_train_hot.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# define the checkpoint
filepath = "./weights-improvement-{epoch:02d}-{loss:.4f}"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
model.fit(x_train, y_train_hot, nb_epoch=20, batch_size=16, callbacks=callbacks_list, validation_split=0.1)

predictions = model.predict_proba(x_test)

file_submission = open("output.csv", "w")
for i, vector in enumerate(predictions):
    file_submission.write(str(int(i+1)) + "," + str(vector[1]) + "\n")
