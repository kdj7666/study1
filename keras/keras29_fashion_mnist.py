# 29 - 100 

from asyncio import to_thread
from json import encoder
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D # 이미지는 2차원 2d
from tensorflow.keras.datasets import mnist
from keras.datasets import fashion_mnist
import numpy as np
import time
import tensorflow as tf 
import pandas as pd
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

# 1 data

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape, y_train.shape)   # ( 50000, 32, 32 3) ( 50000, 1 )
print(x_test.shape, y_test.shape)     # ( 10000, 32, 32 3) ( 10000, 1 )

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

# y_train = pd.get_dummies(y_train)
# y_test = pd.get_dummies(y_test)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


model.add(Conv2D(filters=64, kernel_size=(3,3), input_shape=(28, 28, 1)))

model.add(Conv2D(10, kernel_size=(3,3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (2,2), padding='valid'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.add(Dense(10, activation='softmax'))



# compile. epochs

start_time = time.time()

earlystopping = EarlyStopping(monitor='val_loss', patience=150, mode='min', verbose=1,
                              restore_best_weights=True)

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',
              metrics = ['accuracy'])

a = model.fit(x_train, y_train, epochs=200, batch_size=600,
              validation_split=0.2, callbacks= [earlystopping], verbose=1)

end_time = time.time()-start_time
print(a)
print(a.history['val_loss'])


# # 4. evaluate , predict
# result = model.evaluate(x_test, y_test)
# print('loss : ', result[0])
# print('accuracy : ', result[1])

# loss = model.evaluate(x_test, y_test)
# y_predict = model.predict(x_test)

# y_test = tf.argmax(y_test, axis=1)
# acc = accuracy_score(x_test, y_test)

# print('acc : ', acc)
# print('loss : ', loss)


# y_predict = model.predict(x_test)

loss = model.evaluate(x_test, y_test) 
print('loss : ', loss)

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis= 1)
y_predict = to_categorical(y_predict)


acc = accuracy_score(y_test, y_predict)
print('acc스코어 : ', acc)


# loss: 0.3216 
# - accuracy: 0.8890

# loss: 0.3179 - accuracy: 0.8926


# loss: 0.0688 - accuracy: 0.9887 
# val_loss: 0.5930 - val_accuracy: 0.8838


# loss: 0.6439 
# accuracy: 0.8721
# loss :  [0.6439264416694641, 0.8720999956130981]
# acc스코어 :  0.8721


# - accuracy: 0.8739
# loss :  [0.6156235337257385, 0.8738999962806702]
# acc스코어 :  0.8739