# 평가
# rnn 계열 / cnn계열 필수 rnn 으로  
# 시계열로 잘르기 split 

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
from tensorflow.python.keras.models import Sequential, Input
from tensorflow.python.keras.layers import Dense, Conv1D, Conv2D, LSTM
from tensorflow.python.keras.layers import Flatten, MaxPooling2D
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler

path = './_data/kaggle_jena/'

f = pd.read_csv(path + 'jena_climate_2009_2016.csv',
                index_col=0)   # 인덱스 컬럼 0 으로 인해 jena 파일의 1번줄 
                               # data time 제거 그로인한 shape 값 변경 

print(f)
print(f.shape)   # (420551, 15)  ----  (420551, 14)

print('====================================')

size = 4

def split_x(dataset, size1):
    aaa = []
    for i in range(len(dataset) - size1 + 1):   # 10 - 9 + 1 = 2
        subset = dataset[i : (i + size1)]
        aaa.append(subset)  
    return np.array(aaa)

aaa = split_x(f,size)
print(aaa)
print(aaa.shape)   # (420550, 2, 15) ---- (420550, 2, 14)
print('====================================')
x = aaa[:, :-1]
y = aaa[:, -1]
print('====================================')
# print(x,y)
# print(x.shape, y.shape)   # (420548, 3, 14) (420548, 14)

print(f.info())
# f_set = f.dropna(columns=['Date Time'], axis=1)

print(f.shape)   # (420551, 14)
print('====================================')
print(f.columns)

# x_train, x_test, y_train, y_test = train_test_split(x,y,
#         train_size=0.8, shuffle=True, random_state=55)

print('==========================================================')





model = Sequential()

model.add(LSTM(units=100, input_shape=(3,14)))
model.add(Dense(100, activation='swish'))
model.add(Dense(80, activation='swish'))
model.add(Dense(60, activation='swish'))
model.add(Dense(10, activation='swish'))
model.add(Dense(1))


model.summary()

model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=1500 , batch_size=2000, verbose=1)


loss = model.evaluate(x,y)

print('loss : 의 결과 ', loss)


