import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout, Conv2D, Flatten, LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import nan_euclidean_distances, r2_score, mean_squared_error
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

#1. data  # 10번 경로  +는 문자가 연결이 된다
path = './_data/bike/' 
train_set = pd.read_csv(path + 'train.csv', 
                        index_col=0)         #index_col 1번째는 id 행의 이름이기때문에 계산 ㄴ

test_set = pd.read_csv(path + 'test.csv',
                       index_col=0)     #  예측에서 프레딕트로 쓸것이다

submission = pd.read_csv(path + 'samplesubmission.csv')

print(test_set)
print(test_set.shape)   # 715개의 열과 9개의 컬럼  (715,9)

print(train_set.columns) 
print(train_set.info())       # 컬럼에 대한 내용이 디테일하게 나온다       ( Non-Null Count ) 이빨이 빠졋다 데이터가 빠졋다  [ 결측치 ] 데이터 전처리에 아주 중요 / [이상치]라는 데이터도 있다 나중에 
print(train_set.describe())               #  describe 묘사하다 서술하다  # 최솟값 최댓값 등 확인       pd 좀더 찾아보기 중요

#### 결측치 처리 1. 제거####
print(train_set.isnull().sum())
train_set = train_set.dropna()
print(train_set.isnull().sum())
print(train_set.shape)
#############################
x = train_set.drop(['count','casual','registered',], axis=1)   # drop 날리다 카운트라는 줄을 날릴것이다 소숫점이 1개 
print(x)
print(x.columns)
print(x.shape) #  ( 1459 , 9 )

y = train_set['count']  # 이렇게 하면 빠진다 지금은 이정도 ( [ ] 대괄호를 잘못치면 다 틀린다 ) 나중에 반복
print(y)
print(y.shape)   # ( 1459 , ) # 벡터가 1개 그래서 최종 아웃풋 갯수는 1개   ( 여기까지가 데이터 )

print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.989,
        shuffle=True,
        random_state=40)

print(x_train.shape, x_test.shape) # (10766, 8) (120, 8)
print(y_train.shape, y_test.shape) # (10766,) (120,)

print(np.unique(y_train, return_counts=True))
print(np.unique(y_test, return_counts=True))


# scaler = MinM# from sklearn.preprocessing import MinMaxScaler, StandardScaler
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

x_train = x_train.reshape(10766,8,1)
x_test = x_test.reshape(120,8,1)

# np.logical_or(x, y)
# print(x = train_set.info(x))
# print(train_set.dropna( subset['Age']))
# print(pd.isna('nan'))

#2. 모델구성
model = Sequential()
model.add(LSTM(units=64, return_sequences=True,
               input_shape=(8,1)))
model.add(LSTM(32, return_sequences=False,
               activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()


# model.add(Dense(90, input_dim=8))          # 행 무시 열 우선 필수 
# model.add(Dense(80, activation='relu'))
# model.add(Dense(80, activation='relu'))  
# model.add(Dropout(0.3))
# model.add(Dense(50))
# model.add(Dense(50, activation='relu'))
# model.add(Dropout(0.3))  
# model.add(Dense(50))   
# model.add(Dense(1))

# input1 = Input(shape=(8,))
# dense1 = Dense(10)(input1)
# dense2 = Dense(5, activation='relu')(dense1)
# dense3 = Dense(3, activation='relu')(dense2)
# output1 = Dense(1)(dense3)
# model = Model(inputs=input1, outputs=output1)

# model.save('./_save/k23_smm_bike.h5')

#3. 컴파일 훈련

start_time = time.time()
model.compile(loss='mse', optimizer = 'adam')   
# 평가지표는 프레딕트 결과값 어쩌구 저쩌구 해서 mse 로 가능 비슷하면 된다 

earlystopping = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, # mode='min'뿐아니라 max도 있음  디폴드값 찾아볼것 모르면 오토 
              restore_best_weights=True)
a = model.fit(x_train, y_train, epochs=100, batch_size=500,
          validation_split=0.2,
          callbacks = [earlystopping],
          verbose=1)             

print(a)
print(a.history['val_loss']) # 대괄호로 loss , val loss 값 출력 가능

end_time = time.time()-start_time

#4. 평가 예측

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2scoer :', r2)
print('걸린시간: ', end_time)
model.summary()
# print(y_summit)
# print(y_summit.shape) # (715,1)

######################## .to_csv()를 사용해서 아이디값 안됨 카운트값 순서대로 
### submission.csv를 완성하시오 !!! ( 과제 겸 실습 )
# dataframe = pd.DataFrame(y_summit)
# dataframe.to_csv('.csv')

# submission['count'] = y_summit        
# submission = submission.fillna(submission.mean())
# submission.to_csv(path + 'samplesubmission.csv', index=False)

# loss :  26796.220703125
# r2scoer : 0.36441338064926876
# 걸린시간:  251.71542096138

