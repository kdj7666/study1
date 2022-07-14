import numpy as np
from sklearn.datasets import fetch_covtype
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout, Conv2D, Flatten, Conv1D
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
tf.random.set_seed(66)
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, r2_score 
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import pandas as pd 

#1. data
path = './_data/ddarung/' 
train_set = pd.read_csv(path + 'train.csv',
                    index_col=0)
test_set = pd.read_csv(path + 'test.csv',
                    index_col=0)

submisson = pd.read_csv(path + 'submission.csv',
                        index_col=0)

print(train_set.columns)
print(train_set.info()) #null은 누락된 값이라고 하고 "결측치"라고도 한다.
print(train_set.describe())

print(train_set.isnull().sum())
train_set = train_set.fillna(train_set.mean())
print(train_set.isnull().sum())
print(train_set.shape)
test_set = test_set.fillna(test_set.median())

x = train_set.drop(['count'],axis=1)
# x = test_set 

y = train_set['count']
# y = test_set

print(x.shape, y.shape) # (1459, 9) (1459,)

x_train, x_test, y_train, y_test = train_test_split(x,y, # shuffle True False 잘 써야 한다 
                    test_size=0.2,
                    random_state=30)

print(x_train.shape, x_test.shape) # (1167, 9) (292, 9)
print(y_train.shape, y_test.shape) # (1167,) (292,)

print(np.unique(y_train, return_counts=True))  # (array([0, 1, 2]), array([59, 71, 48], dtype=int64))]
print(np.unique(y_test, return_counts=True))

print('==========================================================')


# 181번

# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from sklearn.preprocessing import MaxAbsScaler, RobustScaler
# scaler = RobustScaler()
# scaler = MaxAbsScaler()

# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_train))   # 0.0
print(np.max(x_train))   # 0.0 컬럼별로 나누어주어야 한다
print(np.min(x_test))
print(np.max(x_test))

x_train = x_train.reshape(1167,3,3)
x_test = x_test.reshape(292,3,3)

print(x_train.shape, x_test.shape)

# model

model = Sequential()
model.add(Conv1D(32, 2, padding='same', input_shape=(3,3)))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1)) 
model.summary()

# input1 = Input(shape=(13,))
# dense1 = Dense(10)(input1)
# dense2 = Dense(5, activation='relu')(dense1)
# dense3 = Dense(3, activation='relu')(dense2)
# output1 = Dense(3, activation='softmax')(dense3)
# model = Model(inputs=input1, outputs=output1)


# compile , epochs 
earlystopping = EarlyStopping(monitor='loss', patience=100, mode='min', verbose=1,
                              restore_best_weights=True)

model.compile(loss = 'mae', optimizer = 'adam')

start_time = time.time()

a = model.fit(x_train, y_train, epochs=300, batch_size=100,
          validation_split=0.2,
          callbacks = [earlystopping],
          verbose=1)

# loss ,acc = model.evaluate(x_test, y_test)


# print('====================================')
# print(y_test[:5])
# print('====================================')

loss = model.evaluate(x_test, y_test)
print('loss : ,', loss)
# y_pred = model.predict(x_test[:5])
# print(y_pred)
print('====================================')
end_time = time.time()-start_time

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('걸린시간 : ', end_time)
print('r2.score:', r2)
model.summary()

# loss :  0.338166207075119
# accuracy :  0.9074074029922485
# acc.score :  0.9074074074074074
# 걸린시간 :  14.029605388641357
# r2.score: 0.8362644026682838

# loss : , 30.37053108215332
# ====================================
# 걸린시간 :  25.235912084579468
# r2.score: 0.7459924007308059



# loss : , 31.725061416625977
# ====================================
# 걸린시간 :  23.225853443145752
# r2.score: 0.7036943534057598


