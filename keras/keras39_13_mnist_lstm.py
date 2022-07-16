# 27 - 2 
from json import encoder
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input, LSTM # 이미지는 2차원 2d
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

x_train = x_train.reshape( 60000, 28*28)
x_test = x_test.reshape( 10000, 28*28)

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

x_train = x_train.reshape( 60000, 28, 28)
x_test = x_test.reshape( 10000, 28, 28)

# 만들어봐 
# acc 0.98 이상 
# Conv2D 3줄 이상 
# onehotencoding 잊지말기

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

# model = Sequential()

# cnn 4차원

# model.add(Conv2D(filters=64,kernel_size=(3,3),
#                 padding='same', input_shape=(28, 28, 1)))
# model.add(Conv2D(10, kernel_size=(3,3)))
# model.add(MaxPooling2D())
# model.add(Conv2D(32, (2,2), padding='valid'))
# model.add(Flatten())
# model.add(Dense(32, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(10, activation='softmax'))

# 함수형 모델 
model = Sequential()

model.add(LSTM(units=10, return_sequences=True,
               input_shape=(28,28)))
model.add(LSTM(10, return_sequences=False,
               activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='linear'))
model.summary()

# compile. epochs

start_time = time.time()

earlystopping = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=1,
                              restore_best_weights=True)

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',
              metrics = ['accuracy'])

a = model.fit(x_train, y_train, epochs=10, batch_size=3000,
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

# acc = accuracy_score(y_test, y_predict)

# print('acc.score : ', acc)
print('걸린시간 : ', end_time)

#   accuracy: 0.9800

#  accuracy: 0.9807

# loss: 0.0539 - accuracy: 0.9849

# loss :  [6.705249786376953, 0.07419999688863754]
# r2score :  -0.1528223553993878

