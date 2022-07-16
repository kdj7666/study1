# 28 - 2
from asyncio import to_thread
from json import encoder
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, LSTM # 이미지는 2차원 2d
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

x_train = x_train.reshape(50000, 32, 96)
x_test = x_test.reshape(10000, 32, 96)
print(x_train.shape)    # (50000, 32, 32, 3)
print(np.unique(x_train, return_counts=True))

# [ One Hot Encoding ]- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
from tensorflow.python.keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape)



model = Sequential()
model.add(LSTM(units=10, return_sequences=True,
               input_shape=(32,96)))
model.add(LSTM(10, return_sequences=False,
               activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()


# compile. epochs

start_time = time.time()

earlystopping = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=1,
                              restore_best_weights=True)

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',
              metrics = ['accuracy'])

a = model.fit(x_train, y_train, epochs=100, batch_size=10000,
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
# print('걸린시간 : ', end_time)

# accuracy: 0.5266

# accuracy: 0.9819

# loss :  [1.6611863374710083, 0.3865000009536743]
# r2score :  0.1843408426817577

# loss :  [1.6673167943954468, 0.39430001378059387]
# r2score :  0.1866898920372996
