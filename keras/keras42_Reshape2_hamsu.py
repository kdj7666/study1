# 27 - 2 
from json import encoder
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D # 이미지는 2차원 2d
from tensorflow.python.keras.layers import Conv1D, LSTM, Reshape, Input
from tensorflow.keras.datasets import mnist
import numpy as np
import time
import tensorflow as tf 
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 1 data

(x_train, y_train), (x_test, y_test) = mnist.load_data()


print(x_train.shape, y_train.shape)   # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)     # (10000, 28, 28) (10000,)

x_train = x_train.reshape( 60000, 28*28*1 )
x_test = x_test.reshape( 10000, 28*28*1 )

print(x_train.shape)      # (60000, 28, 28, 1)
print(y_train.shape)

print(np.unique(y_train, return_counts=True))
print(np.unique(y_test, return_counts=True))

# scaler = RobustScaler()
scaler = StandardScaler()
scaler.fit(x_train)
scaler.fit(x_test)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_train))   # 0.0
print(np.max(x_train))   # 0.0 컬럼별로 나누어주어야 한다
print(np.min(x_test))
print(np.max(x_test))

x_train = x_train.reshape( 60000, 28, 28, 1 )
x_test = x_test.reshape( 10000, 28, 28, 1 )

# 만들어봐 
# acc 0.98 이상 
# Conv2D 3줄 이상 
# onehotencoding 잊지말기

model = Sequential()

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)



model = Sequential()
input1 = Input(shape=(28,28,1))                                        # n, 28, 28, 1
dense1 = Conv2D(filters=64, kernel_size=(3,3), padding='same')(input1) # n, 28 28 64
dense2 = MaxPooling2D(2,2)(dense1)                                     # n, 14 14 64
dense3 = Conv2D(32, (4,4), padding='valid')(dense2)                    # n, 11 11 32
dense4 = MaxPooling2D(2,2)(dense3)                                     # n, 5 5 32
dense5 = Conv2D(16, (4,4), padding='valid')(dense4)                    # n, 2 2 16
dense6 = Reshape(target_shape=(64,))(dense5)                           # n, 4, 64 
dense7 = Dense(4, activation='relu')(dense6)
dense8 = Dense(10, activation='relu')(dense7)
output1 = Dense(10, activation='softmax')(dense8)
model = Model(inputs=input1, outputs=output1)
model.summary()




# model.add(Conv2D(filters=64,kernel_size=(3,3),
#                 padding='same', input_shape=(28, 28, 1)))
# model.add(MaxPooling2D()) # 전체크기를 반으로 줄이고 좋은 값만 찾는다 ( 14 14 64)
# model.add(Conv2D(32, (3,3)))      # (n, 12, 12, 32)
# model.add(Conv2D(7, (3,3)))       # (n, 10, 10, 7)
# model.add(Reshape(target_shape=(100,7)))   # (n, 32, 1)   # 모양만 바뀐다 데이터조작x 순서,내용 바뀌지않음 (연산안됨)
# model.add(Conv1D(10, kernel_size=3))        # (n, 10, 3)
# model.add(LSTM(16))                        # (n, 16, 1)  rnn 모델 LSTM 3차원 이기 때문에 3차원으로 받아야함 
# model.add(Reshape(target_shape=(16,1)))    # (n, 16, 1)  
# model.add(Dense(32, activation='relu'))   # (n, 32)
# model.add(Dense(10, activation='softmax')) # 10은 units  # (n, 10)



# conv숫자d 차원이 다 같다 입력하는것 -> 출력하는것
# conv2d  4차원 -> 4차원
# conv1d  3차원 -> 3차원 



# model.add(Conv2D(filters=64,kernel_size=(3,3),
#                 padding='same', input_shape=(28, 28, 1)))
# model.add(MaxPooling2D()) # 전체크기를 반으로 줄이고 좋은 값만 찾는다 ( 14 14 64)
# model.add(Conv2D(32, (3,3)))      # (n, 12, 12, 32)
# model.add(Conv2D(7, (3,3)))       # (n, 10, 10, 7)
# model.add(Flatten())              # (n, 700)    # 모양만 바뀐다 데이터조작x 순서,내용 바뀌지않음   (연산안됨)
# model.add(Dense(100, activation='relu'))   # (n, 100) 
# model.add(Reshape(target_shape=(100,1)))   # (n, 100, 1)   # 모양만 바뀐다 데이터조작x 순서,내용 바뀌지않음 (연산안됨)
# model.add(Conv1D(10, kernel_size=3))                  # (n, 10, 3)
# model.add(LSTM(16))                        # (n, 16)
# model.add(Dense(32, activation='relu'))   # (n, 32)
# model.add(Dense(10, activation='softmax')) # 10은 units  # (n, 10)
# # conv숫자d 차원이 다 같다 입력하는것 = 출력하는것

# model.summary()



# compile. epochs

start_time = time.time()

earlystopping = EarlyStopping(monitor='val_loss', patience=150, mode='min', verbose=1,
                              restore_best_weights=True)

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',
              metrics = ['accuracy'])

a = model.fit(x_train, y_train, epochs=200, batch_size=3000,
              validation_split=0.2, callbacks= [earlystopping], verbose=1)

end_time = time.time()-start_time
print(a)
print(a.history['val_loss'])


# 4. evaluate , predict

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2score : ', r2)
acc = accuracy_score(y_test, y_predict)

print('acc.score : ', acc)
print('걸린시간 : ', end_time)

#   accuracy: 0.9800

#  accuracy: 0.9807


# loss :  [0.22978374361991882, 0.9607999920845032]
# r2score :  0.9273070749609195


# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_1 (InputLayer)         [(None, 28, 28, 1)]       0
# _________________________________________________________________
# conv2d (Conv2D)              (None, 28, 28, 64)        640
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) (None, 14, 14, 64)        0
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 11, 11, 32)        32800
# _________________________________________________________________
# max_pooling2d_1 (MaxPooling2 (None, 5, 5, 32)          0
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 2, 2, 16)          8208
# _________________________________________________________________
# flatten (Flatten)            (None, 64)                0
# _________________________________________________________________
# dense (Dense)                (None, 30)                1950
# _________________________________________________________________
# dense_1 (Dense)              (None, 30)                930
# _________________________________________________________________
# dense_2 (Dense)              (None, 10)                310
# =================================================================
# Total params: 44,838
# Trainable params: 44,838
# Non-trainable params: 0

# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_1 (InputLayer)         [(None, 28, 28, 1)]       0
# _________________________________________________________________
# conv2d (Conv2D)              (None, 28, 28, 64)        640
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) (None, 14, 14, 64)        0
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 11, 11, 32)        32800
# _________________________________________________________________
# max_pooling2d_1 (MaxPooling2 (None, 5, 5, 32)          0
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 2, 2, 16)          8208
# _________________________________________________________________
# reshape (Reshape)            (None, 4, 16)             0
# _________________________________________________________________
# dense (Dense)                (None, 4, 30)             510
# _________________________________________________________________
# dense_1 (Dense)              (None, 4, 30)             930
# _________________________________________________________________
# dense_2 (Dense)              (None, 4, 10)             310
# =================================================================
# Total params: 43,398
# Trainable params: 43,398
# Non-trainable params: 0

