# 컬럼은 최소 7개 거래량 은 반드시 들어갈것
# 아모레 가격을 맞춘다 화요일 시가 수요일 종가 
# 삼성 , 아모레 앙상블 해서 아모레 값을 찾기 
# LSTM 

from datetime import datetime
from random import random
import numpy as np
import pandas as pd
import time
import os
from requests import head
from tensorflow.python.keras.models import Sequential, Model, Input
from tensorflow.python.keras.layers import Dense, Conv1D, Conv2D, LSTM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import concatenate, Concatenate

path = './_data/test_amore_0718/'
df1 = pd.read_csv(path + '아모레220718.csv', thousands=",", encoding='cp949')
df2 = pd.read_csv(path + '삼성전자220718.csv', thousands=",", encoding='cp949')


print(df1.shape, df2.shape) # (3180, 17) (3040, 17)
print('=======================================================')
print(df1.columns, df2.columns)  #Index(['일자', '시가', '고가', '저가', '종가', '전일비', 'Unnamed: 6', '등락률', '거래량',
    #                                    '금액(백만)', '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'],
    #
    #             dtype='object') Index(['일자', '시가', '고가', '저가', '종가', '전일비', 'Unnamed: 6', '등락률', '거래량',
    #                                    '금액(백만)', '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'],
    #                                    dtype='object')


print(df1.info())
print('==========================================================')
print(df2.info())


df1['일자'] = pd.to_datetime(df1['일자'], infer_datetime_format=True)
df1['연도'] =df1['일자'].dt.year
df1['월'] =df1['일자'].dt.month
df1['일'] =df1['일자'].dt.day

df2['일자'] = pd.to_datetime(df2['일자'], infer_datetime_format=True)
df2['연도'] =df2['일자'].dt.year
df2['월'] =df2['일자'].dt.month
df2['일'] =df2['일자'].dt.day

df1 = df1.loc[df1['일자']>="2018/05/04"]
df2 = df2.loc[df2['일자']>="2018/05/04"]
print(df1.shape, df2.shape) # (1035, 20) (1035, 20)

df1 = df1.sort_values(by=['일자'], axis=0, ascending=True) # 오름차순으로 정렬
df2 = df2.sort_values(by=['일자'], axis=0, ascending=True)


print(df1.head(5))
print(df2.head(5))

# df1 = df1.drop(['Date_Time'], axis=1)
# df2 = df2.drop(['Date_Time'], axis=1)

df1 = df1.drop(['일자','신용비','기관','외국계','프로그램','외인비','개인','금액(백만)','외인(수량)','전일비','Unnamed: 6'], axis=1)
df2 = df2.drop(['일자','신용비','기관','외국계','프로그램','외인비','개인','금액(백만)','외인(수량)','전일비','Unnamed: 6'], axis=1)

print(df1.shape, df1.columns)  # 3180, 10
print('====================================')
print(df2.shape, df2.columns)  # 3040, 10 ( 시가 고가 저가 종가 등락률 거래량 연도 월 일 )

# df1['전일비'] = df1['전일비'].astype(int)
# df2['전일비'] = df2['전일비'].astype(int)

# df = df1.convert_objects(convert_numeric=True)
# df = df2.convert_objects(convert_numeric=True)

print(df1.info())
print(df2.info())

x1 = df1
x2 = df2

print(x1.shape, x2.shape) # (1035, 9) (1035, 9)
print('==========================================================')
print('==========================================================')
print('==========================================================')

y1 = df1['종가']


# split

size1 = 5

def split_x(df, size1):
    aaa = []
    for i in range(len(df) - size1 + 1):
        subset = df[i : (i + size1)]
        aaa.append(subset)  
    return np.array(aaa)

bbb = split_x(x1, size1)
ccc = split_x(x2, size1)
ddd = split_x(y1, size1)

print(bbb)
print(bbb.shape) # (3176, 5, 9)
print('===================================')

x1 = bbb[:,:-3]
x2 = ccc[:,:-3]
y = ddd[:,-3:]

print(y, y.shape)


print(x1.shape, y.shape) # (1031, 2, 9) (1031, 3)

print('===================================')

print(x2.shape, y.shape) # (1031, 2, 9) (1031, 3)

x1_train, x1_test, x2_train, x2_test, y_train, y_test= train_test_split(x1,x2,y,
                    train_size=0.88, shuffle=False)


# x1_train, x1_test, y1_train, y1_test = train_test_split(x1,y1,
#                                         train_size=0.7, shuffle=False)
# x2_train, x2_test, y2_train, y2_test = train_test_split(x2,y2,
#                                         train_size=0.7, shuffle=False)

# x1_train = x1_train.reshape(699,5, 9)
# x2_train = x2_train.reshape(699,5, 9)

# x1_test = x1_test.reshape(300,5, 9)
# x2_test = x2_test.reshape(300,5, 9)


model = Sequential()

input1 = Input(shape=(2, 9))
dense1 = Conv1D(64, 2, activation='relu', name='d1')(input1)
dense2 = LSTM(64, activation='relu', name='d2')(dense1)
dense3 = Dense(64, activation='relu', name='d3')(dense2)
output1 = Dense(32, activation='relu', name='out_d1')(dense3)

# 2-2. 모델2
input2 = Input(shape=(2, 9))
dense11 = Conv1D(64, 2, activation='relu', name='d11')(input2)
dense12 = LSTM(64, activation='swish', name='d12')(dense11)
dense13 = Dense(64, activation='relu', name='d13')(dense12)
dense14 = Dense(32, activation='relu', name='d14')(dense13)
output2 = Dense(16, activation='relu', name='out_d2')(dense14)

from tensorflow.python.keras.layers import concatenate

merge1 = concatenate([output1, output2], name='m1')
merge2 = Dense(100, activation='relu', name='mg2')(merge1)
merge3 = Dense(100, name='mg3')(merge2)
merge4 = Dense(50, name='mg4')(merge3)
last_output = Dense(1, name='output4')(merge4)

model = Model(inputs=[input1, input2], outputs=[last_output])
# model.summary()

from tensorflow.python.keras.callbacks import EarlyStopping
start_time = time.time()

model.compile(loss = 'mse', optimizer = 'adam')

earlystopping = EarlyStopping(monitor='val_loss', patience=100, mode='auto', verbose=1,
                              restore_best_weights=True)


a = model.fit([x1_train, x2_train],y_train, epochs=220, batch_size=30,
              validation_split=0.2,
              callbacks= [earlystopping], verbose=1)

end_time = time.time()-start_time
print(a)
print(a.history['val_loss'])

# 4. evaluate , perdict

loss = model.evaluate([x1_test,x2_test], y_test)
predict= model.predict([x1_test,x2_test])

# loss = model.evaluate([x1_test, x2_test], y_test)
# predict = model.predict([x1_test, x2_test])
print('loss: ', loss)
print('predict: ', predict[-1:])
print('걸린 시간: ', end_time)


