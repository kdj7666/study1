# 28 - 2
from asyncio import to_thread
from json import encoder
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, LSTM, Conv1D # 이미지는 2차원 2d
from tensorflow.keras.datasets import mnist
from keras.datasets import cifar10
import numpy as np
import time
import tensorflow as tf 
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

# 1 data

(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# loss의 스케일 조정을 위해 0 ~ 255 -> 0 ~ 1 범위로 만들어줌
x_train = x_train.astype('float32')  # astype() 메서드는 계열의 값을 int 유형에서 float 유형으로 변환하는 데 사용됩니다
x_test = x_test.astype('float32')
x_train = x_train/255
x_test = x_test/255

mean = np.mean(x_train, axis=(0 , 1 , 2 , 3))
std = np.std(x_train, axis=(0 , 1 , 2 , 3))
x_train = (x_train-mean)/std
x_test = (x_test-mean)/std

x_train = x_train.reshape(50000, 1024, 3)
x_test = x_test.reshape(10000, 1024, 3)
print(x_train.shape)    # (50000, 32, 32, 3)
print(np.unique(x_train, return_counts=True))

# [ One Hot Encoding ]- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
from tensorflow.python.keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape)



model = Sequential()
model.add(Conv1D(32, 2, padding='same', input_shape=(1024,3)))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()


# compile. epochs

start_time = time.time()

earlystopping = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=1,
                              restore_best_weights=True)

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',
              metrics = ['accuracy'])

a = model.fit(x_train, y_train, epochs=10, batch_size=1000,
              validation_split=0.2,
              callbacks= [earlystopping], verbose=1)

end_time = time.time()-start_time
print(a)
print(a.history['val_loss'])


# 4. evaluate , predict

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2score : ', r2)
# acc = accuracy_score(y_test, y_predict)

# print('acc.score : ', acc)
print('걸린시간 : ', end_time)

# accuracy: 0.5266

# accuracy: 0.9819


# loss :  [1.6026703119277954, 0.4415999948978424]
# r2score :  0.22603455161627525
# 걸린시간 :  12.653355836868286