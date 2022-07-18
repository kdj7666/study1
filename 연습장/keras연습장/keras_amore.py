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

path = './_data/test_amore_0718/'
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

df1 = df1.sort_values(by=['일자'], axis=0, ascending=False) # 오름차순으로 정렬
df2 = df2.sort_values(by=['일자'], axis=0, ascending=False)


print(df1.head(5))
print(df2.head(5))

print('====================================')
print(df1.shape, df1.columns)
print('====================================')
print(df2.shape, df2.columns)
print('====================================')
# df1 = df1.drop(['Date_Time'], axis=1)
# df2 = df2.drop(['Date_Time'], axis=1)


df1 = df1.drop(['전일비','일자','Unnamed: 6','신용비','기관','외국계','프로그램','외인비','고가','저가','종가'], axis=1)
df2 = df2.drop(['전일비','일자','Unnamed: 6','신용비','기관','외국계','프로그램','외인비','고가','저가','종가'], axis=1)

print(df1.shape, df1.columns)  # 3180, 9
print('====================================')
print(df2.shape, df2.columns)  # 3040, 9

x1 = df1
x2 = df2

print(x1.shape, x2.shape) # (1035, 9) (1035, 9)
print('==========================================================')
print('==========================================================')
print('==========================================================')

y1 = df1['시가']
y2 = df2['시가']

# split 


size1 = 5

def split_x(df1, size1):
    aaa = []
    for i in range(len(df1) - size1 + 1):
        subset = df1[i : (i + size1)]
        aaa.append(subset)  
    return np.array(aaa)


def split_g(df2, size1):
    rrr = []
    for l in range(len(df2) - size1 + 1):
        subset = df2[l : (l + size1)]
        rrr.append(subset)  
    return np.array(rrr)

bbb = split_x(x1, size1)
print(bbb)
print(bbb.shape) # (3176, 5, 9)
print('===================================')

x1 = bbb[:, :-1]
y = bbb[:, -1]

print(x1.shape, y1.shape) # (999, 5, 9) (999, 5, 9)
print('===================================')
qqq = split_g(x2, size1)
print(qqq) 
print('===================================')


x2 = qqq[:, :-1]
y = qqq[:, -1]


print('===================================')
print('===================================')


print(x2.shape, y2.shape)  # (999, 5, 9) (999, 5, 9)

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1,x2,y,
                    train_size=0.8, shuffle=False)

# x1_train, x1_test, y1_train, y1_test = train_test_split(x1,y1,
#                                         train_size=0.7, shuffle=False)
# x2_train, x2_test, y2_train, y2_test = train_test_split(x2,y2,
#                                         train_size=0.7, shuffle=False)
print(x1_train.shape, y_train.shape)
print(x2_train.shape, y_train.shape)

print(x1_test.shape, y_test.shape)
print(x2_test.shape, y_test.shape)


# x1_train = x1_train.reshape(699,5, 9)
# x2_train = x2_train.reshape(699,5, 9)

# x1_test = x1_test.reshape(300,5, 9)
# x2_test = x2_test.reshape(300,5, 9)



model = Sequential()

input1 = LSTM(units=10, input_shape=(5,9))
dense1 = Dense(10, activation='relu', name='dj1')(input1)
dense2 = Dense(10, activation='relu', name='dj2')(dense1)
dense3 = Dense(10, activation='relu', name='dj3')(dense2)
output1 = Dense(10, activation='relu', name='out_dj1')(dense3)


input2 = LSTM(units=10, input_shape=(5,9))
dense11 = Dense(10, activation='relu', name='dj11')(input2)
dense12 = Dense(10, activation='relu', name='dj12')(dense11)
dense13 = Dense(10, activation='relu', name='dj13')(dense12)
dense14 = Dense(10, activation='relu', name='dj14')(dense13)
output2 = Dense(10, activation='relu', name='out_dj2')(dense14)


merge1 = concatenate([output1, output2], name='mg1') # 두개의 아웃풋이 합쳐진 하나의 레이어 층 
merge2 = Dense(15, activation='relu', name='mg2')(merge1)
merge3 = Dense(10, name='mg3')(merge2)
merge4 = Dense(5, name='mg4')(merge3)
last_output = Dense(1, name='last')(merge4)

model = Model(inputs=[input1, input2],outputs=last_output) # 모델의 정의가 됨



from tensorflow.python.keras.callbacks import EarlyStopping
start_time = time.time()

earlystopping = EarlyStopping(monitor='val_loss', patience=200, mode='min', verbose=1,
                              restore_best_weights=True)

model.compile(loss = 'mae', optimizer = 'adam',
              metrics = ['accuracy'])

a = model.fit([x1, x2], epochs=50, batch_size=200,
              validation_split=0.2,
              callbacks= [earlystopping], verbose=1)

end_time = time.time()-start_time
print(a)
print(a.history['val_loss'])

# 4. evaluate , perdict

loss = model.evaluate([x1,x2],y1)
print('loss : ', loss)
y_predict= model.predict([x1,x2])

from sklearn.metrics import r2_score

r2_2 = r2_score(y1, y_predict)

print('r2score : ', r2_2)
print('loss : ', loss)
print('걸린시간 : ', end_time)
print('keras46.py')

