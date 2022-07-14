import numpy as np
from sklearn.datasets import load_wine
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout, Conv2D, Flatten, LSTM
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
tf.random.set_seed(66)
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, r2_score 
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler


#1. data
datasets = load_wine()
x = datasets.data
y = datasets.target


print(x.shape)
x = x.reshape(178,13,1)



print(np.unique(y, return_counts=True))  # (array([0, 1, 2]), array([59, 71, 48], dtype=int64))]

x_train, x_test, y_train, y_test = train_test_split(x,y,
                    train_size=0.7,
                    shuffle=True, random_state=55) # shuffle True False 잘 써야 한다 

print(x_train.shape, x_test.shape) # (124, 13) (54, 13)
print(y_train.shape, y_test.shape) # (124,) (54,)

print(np.unique(y_train, return_counts=True))
print(np.unique(y_test, return_counts=True))

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

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


# model 

model = Sequential()
model.add(LSTM(units=64, return_sequences=True,
               input_shape=(13,1)))
model.add(LSTM(32, return_sequences=False,
               activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()

# input1 = Input(shape=(13,))
# dense1 = Dense(10)(input1)
# dense2 = Dense(5, activation='relu')(dense1)
# dense3 = Dense(3, activation='relu')(dense2)
# output1 = Dense(3, activation='softmax')(dense3)
# model = Model(inputs=input1, outputs=output1)


# compile , epochs 
earlystopping = EarlyStopping(monitor='val_loss', patience=100, mode='auto', verbose=1,
                              restore_best_weights=True)

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',
              metrics = ['accuracy', 'mse'])

start_time = time.time()

a = model.fit(x_train, y_train, epochs=100, batch_size=1000,
          validation_split=0.2,
          callbacks = [earlystopping],
          verbose=1)

# loss ,acc = model.evaluate(x_test, y_test)

results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('accuracy : ', results[1])

# print('====================================')
# print(y_test[:5])
# print('====================================')


# y_pred = model.predict(x_test[:5])
# print(y_pred)
print('====================================')
end_time = time.time()-start_time

y_predict = model.predict(x_test)
# y_predict = np.argmax(y_predict, axis=1)
# print(y_predict)
# y_test = np.argmax(y_test, axis=1)
# print(y_test)
r2 = r2_score(y_test, y_predict)
# acc = accuracy_score(y_test, y_predict)
# print('acc.score : ', acc)
print('걸린시간 : ', end_time)
print('r2.score:', r2)
model.summary()

# loss :  0.338166207075119
# accuracy :  0.9074074029922485
# acc.score :  0.9074074074074074
# 걸린시간 :  14.029605388641357
# r2.score: 0.8362644026682838


# loss :  1.3024721567944653e-07
# accuracy :  0.24074074625968933
# ====================================
# 걸린시간 :  29.492116451263428
# r2.score: -2.1681597498040848