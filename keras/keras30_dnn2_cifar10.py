# 28 - 2
from asyncio import to_thread
from json import encoder
from matplotlib import image
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout # 이미지는 2차원 2d
from tensorflow.keras.datasets import mnist
from keras.datasets import cifar10
import numpy as np
import time
import tensorflow as tf 
import pandas as pd
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# 1 data

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0


print(x_train.shape, y_train.shape)   # ( 50000, 32, 32 3) ( 50000, 1 ) # 열 의 수열은 값만 같으면 바꾸어도 된다 [ 데이터는 건들면 안된다 ]
print(x_test.shape, y_test.shape)     # ( 10000, 32, 32 3) ( 10000, 1 )

x_train = x_train.reshape(50000,3072)
x_test = x_test.reshape(10000,3072)

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
# y_train = pd.get_dummies(y_train)
# y_test = pd.get_dummies(y_test)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()
# model.add(Dense(75,input_shape=(32*32*3)))
model.add(Dense(62, input_shape=(3072,)))
model.add(Dense(130,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(170,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(130,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(250,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(155,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(10,activation='softmax'))

# compile. epochs

start_time = time.time()

earlystopping = EarlyStopping(monitor='val_loss', patience=300, mode='min', verbose=1,
                              restore_best_weights=True)
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',
              metrics = ['accuracy'])
a = model.fit(x_train, y_train, epochs=100, batch_size=32,
              validation_split=0.5, callbacks= [earlystopping], verbose=2)
end_time = time.time()-start_time
print(a)
print(a.history['val_loss'])


# 4. evaluate , predict

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

# r2 = r2_score(y_test, y_predict)
# print('r2score : ', r2)

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
print(y_predict)
y_test = np.argmax(y_test, axis=1)
print(y_test)

acc = accuracy_score(y_test, y_predict)
print('acc.score : ', acc)
print('걸린시간 : ', end_time)



# accuracy: 0.4822

# acc.score :  0.508
