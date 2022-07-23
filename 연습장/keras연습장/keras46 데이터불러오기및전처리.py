'''
# 컬럼은 최소 7개 거래량 은 반드시 들어갈것
# 아모레 가격을 맞춘다 화요일 시가 수요일 종가 
# 삼성 , 아모레 앙상블 해서 아모레 값을 찾기 
# LSTM 
import numpy as np
import pandas as pd
import time
import os
from tensorflow.python.keras.models import Sequential, Model, Input
from tensorflow.python.keras.layers import Dense, Conv1D, Conv2D

path = './_data/test_amore_0718/'

df1 = pd.read_csv(path + '아모레220718.csv', thousands=",", encoding='cp949')

path = './_data/test_amore_0718/'

df2 = pd.read_csv(path + '삼성전자220718.csv', thousands=",", encoding='cp949')


# fname = './test_amore_0718/아모레220718.csv'
# fname_change=""
# for c in fname:
#  if ord(c) <=255:
#   fname_change+=c

# pd = pd.read_csv(fname_change)

# x1_datsets=pd.read_csv("C:\test_amore_0718/아모레220718.csv")


# a = pd.read_csv('./test_amore_0718/삼성전자220718.csv')



# for f in os.listdir(u's:/test_amore_0718/아모레220718.csv'):
#     print(f)



# ab = pd.read_csv(path + 'dkahfp_220718.csv',
#                 index_col=0)

# cd = pd.read_csv(path + 'tkatjdwjswk_220718.csv',
#                 index_col=0)

# print(x1_datsets.shape) # , cd.shape)
print(df1.shape, df2.shape)
 

'''
# 컬럼은 최소 7개 거래량 은 반드시 들어갈것
# 아모레 가격을 맞춘다 화요일 시가 수요일 종가 
# 삼성 , 아모레 앙상블 해서 아모레 값을 찾기 
# LSTM 

from random import random
import numpy as np
import pandas as pd
import time
import os
from tensorflow.python.keras.models import Sequential, Model, Input
from tensorflow.python.keras.layers import Dense, Conv1D, Conv2D, LSTM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from tensorflow.python.keras.callbacks import EarlyStopping

path = './_data/test_amore_0718/'
train_set = pd.read_csv(path + '아모레220718.csv', thousands=",", encoding='cp949')

path = './_data/test_amore_0718/'
test_set = pd.read_csv(path + '삼성전자220718.csv', thousands=",", encoding='cp949')


print(train_set.shape, test_set.shape) # (3180, 17) (3040, 17)
print('=======================================================')
print(train_set.columns, test_set.columns)  #Index(['일자', '시가', '고가', '저가', '종가', '전일비', 'Unnamed: 6', '등락률', '거래량',
    #                                    '금액(백만)', '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'],
    #
    #             dtype='object') Index(['일자', '시가', '고가', '저가', '종가', '전일비', 'Unnamed: 6', '등락률', '거래량',
    #                                    '금액(백만)', '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'],
    #                                    dtype='object')

print(train_set.info())
print('==========================================================')
print(test_set.info())

df_smaple = train_set.loc[:, ['등락률','거래량','금액(백만)','신용비']]

x = train_set.drop(['Date_Time'], axis=1)
y = test_set.drop(['Date_Time'], axis=1)

x = train_set.drop(['전일비'], axis=1)
y = test_set.drop(['전일비'], axis=1)

x = train_set.drop(['Unnamed: 6'], axis=1)
y = test_set.drop(['Unnamed: 6'], axis=1)


data= x.dropna()
data.isnull().sum()
data= y.dropna()
data.isnull().sum()

# scaler = RobustScaler()
# scaler = MaxAbsScaler()

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler.fit(x_train)

# x_train = scaler.transform(df1)
# x_test = scaler.transform(df1)
# x_train = scaler.transform(df2)
# x_test = scaler.transform(df2)
# print(np.min(x_train))   # 0.0
# print(np.max(x_train))   # 0.0 컬럼별로 나누어주어야 한다
# print(np.min(x_test))
# print(np.max(x_test))

print(x.shape, y.shape)

x_train, y_train, x_test, y_test = train_test_split(x,y,
                                                    train_size=0.8)

model = Sequential()
model.add(LSTM(units=100, input_shape=(3,17)))
model.add(Dense(100, activation='swish'))
model.add(Dense(80, activation='swish'))
model.add(Dense(60, activation='swish'))
model.add(Dense(10, activation='swish'))
model.add(Dense(1))

model.summary()


# 맹그러봐
from tensorflow.python.keras.callbacks import EarlyStopping
start_time = time.time()

earlystopping = EarlyStopping(monitor='val_loss', patience=200, mode='min', verbose=1,
                              restore_best_weights=True)

model.compile(loss = 'mse', optimizer = 'adam',
              metrics = ['accuracy'])

a = model.fit(x_train,y_train, epochs=2250, batch_size=200,
              validation_split=0.2,
              callbacks= [earlystopping], verbose=1)

end_time = time.time()-start_time
print(a)
print(a.history['val_loss'])

# 4. evaluate , perdict

loss = model.evaluate(x,y)
print('loss : ', loss)
y_predict= model.predict(x_test)

from sklearn.metrics import r2_score

r2_2 = r2_score(x_test, y_predict)

print('r2score : ', r2_2)
print('loss : ', loss)
print('걸린시간 : ', end_time)
print('keras46.py')
