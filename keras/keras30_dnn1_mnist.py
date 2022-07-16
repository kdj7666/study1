from keras.datasets import mnist
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import time
# [ 실습 ] 완성하시오
# 성능은 cnn보다 좋게 


# 1.1 data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)  # (60000, 28, 28)
print(x_test.shape, y_test.shape)   # (10000, 28, 28)

x_train = x_train.reshape( 60000, 784 )
x_test = x_test.reshape ( 10000 , 784 )

print(np.unique(y_train, return_counts=True))
print(np.unique(y_test, return_counts=True))

scaler = StandardScaler()
scaler.fit(x_train)
scaler.fit(x_test)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_train))
print(np.max(x_train))
print(np.min(x_test))
print(np.max(x_test))

x_train = x_train.reshape(60000,784)
x_test = x_test.reshape(10000,784)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. model
model = Sequential()
# ( model.add(Dense))
# model.add(Dense(64, inpot_shape=(28*28,1 ))) #  두가지 다 가능 # dnn 2차원  순차적
model.add(Dense(64,input_shape=(784,)))
model.add(Dense(61,activation='relu'))
model.add(Dense(37,activation='relu'))
model.add(Dense(54,activation='relu'))
model.add(Dense(86,activation='relu'))
model.add(Dense(17,activation='relu')) 
model.add(Dense(10,activation='softmax'))

# compile, epochs

start_time = time.time()

earlystopping = EarlyStopping(monitor='val_loss', patience=150, mode='min', verbose=1,
                              restore_best_weights=True)

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',
              metrics = ['accuracy'])

a = model.fit(x_train, y_train, epochs=400, batch_size=300,
              validation_split=0.2, callbacks= [earlystopping], verbose=1)

end_time = time.time()-start_time
print(a)
print(a.history['val_loss'])


# 4. evaluate , predict

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis= 1)
y_predict = to_categorical(y_predict)


acc = accuracy_score(y_test, y_predict)
print('acc스코어 : ', acc)


# accuracy: 0.9628

