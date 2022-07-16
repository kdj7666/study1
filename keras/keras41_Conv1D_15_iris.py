from cProfile import label
import numpy as np 
from sklearn.datasets import load_iris
import tensorflow as tf
tf.random.set_seed(66)
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score 
from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout, Conv2D, Flatten, Conv1D
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, accuracy_score
#1. data

datasets = load_iris()

x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x,y,
                    train_size=0.7,
                    shuffle=True, random_state=55) # shuffle True False 잘 써야 한다 

print(x_train.shape, x_test.shape) # (105, 4) (45, 4)
print(y_train.shape, y_test.shape) # (105,) (45,)

x_train = x_train.reshape(105,4,1)
x_test = x_test.reshape(45,4,1)

print(np.unique(y_train, return_counts=True))
print(np.unique(y_test, return_counts=True))


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from sklearn.preprocessing import MaxAbsScaler, RobustScaler
# scaler = RobustScaler()
# scaler = MaxAbsScaler()

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# print(np.min(x_train))   # 0.0
# print(np.max(x_train))   # 0.0 컬럼별로 나누어주어야 한다
# print(np.min(x_test))
# print(np.max(x_test))


# input1 = Input(shape=(4,))
# dense1 = Dense(10)(input1)
# dense2 = Dense(5, activation='relu')(dense1)
# drop1 = Dropout(0.1)(dense2)
# dense3 = Dense(3, activation='relu')(drop1)
# output1 = Dense(3, activation='softmax')(dense3)
# model = Model(inputs=input1, outputs=output1)

model = Sequential()
model.add(Conv1D(32, 2, padding='same', input_shape=(4,1)))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.summary()

earlystopping = EarlyStopping(monitor='val_loss', patience=100, mode='auto', verbose=1,
                              restore_best_weights=True)

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',
              metrics = ['accuracy', 'mse'])

start_time = time.time()

a = model.fit(x_train, y_train, epochs=300, batch_size=100,
          validation_split=0.2,
          callbacks = [earlystopping],
          verbose=1)

# loss ,acc = model.evaluate(x_test, y_test)

# print('loss : ', loss)
# print('accuracy : ', acc)
end_time = time.time()-start_time

results = model.evaluate(x_test, y_test)
# print('loss : ', results[0])
# print('accuracy : ', results[1])

# print('====================================')
# print(y_test[:5])
# print('====================================')

# y_pred = model.predict(x_test[:5])
# print(y_pred)
print('====================================')


y_predict = model.predict(x_test)
# y_predict = np.argmax(y_predict, axis=1)
# print(y_predict)
# y_test = np.argmax(y_test, axis=1)
# print(y_test)
r2 = r2_score(y_test,y_predict)
# acc = accuracy_score(y_test, y_predict)
# print('acc.score : ', acc)
print('걸린시간 : ', end_time)
print('r2.score:', r2)
model.summary()

# acc.score :  0.9333333333333333
# 걸린시간 :  10.506933689117432
# r2.score: 0.8964723926380368

# loss :  0.40474167466163635
# accuracy :  0.9333333373069763

# 걸린시간 :  13.221760272979736
# r2.score: 0.8187775083433945

