# 데이콘 따릉이 문제풀이 
import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import nan_euclidean_distances, r2_score, mean_squared_error
from pandas import DataFrame
import time

#1. data  # 10번 경로  +는 문자가 연결이 된다
path = './_data/ddarung/' 
train_set = pd.read_csv(path + 'train.csv', 
                        index_col=0)         #index_col 1번째는 id 행의 이름이기때문에 계산 ㄴ
print(train_set)             
print(train_set.shape)       # 1459개의 열과 10개의 컬럼  (1459,10)

test_set = pd.read_csv(path + 'test.csv',
                       index_col=0)     #  예측에서 프레딕트로 쓸것이다
print(test_set)
print(test_set.shape)   # 715개의 열과 9개의 컬럼  (715,9)

print(train_set.columns) 
print(train_set.info())       # 컬럼에 대한 내용이 디테일하게 나온다                ( Non-Null Count ) 이빨이 빠졋다 데이터가 빠졋다  [ 결측치 ] 데이터 전처리에 아주 중요 / [이상치]라는 데이터도 있다 나중에 
print(train_set.describe())               #  describe 묘사하다 서술하다  # 최솟값 최댓값 등 확인       pd 좀더 찾아보기 중요

#### 결측치 처리 1. 제거####
print(train_set.isnull().sum())
train_set = train_set.dropna()
print(train_set.isnull().sum())
print(train_set.shape)
#############################


x = train_set.drop(['count'], axis=1)   # drop 날리다 카운트라는 줄을 날릴것이다 소숫점이 1개 
print(x)
print(x.columns)
print(x.shape) #  ( 1459 , 9 )

y = train_set['count']  # 이렇게 하면 빠진다 지금은 이정도 ( [ ] 대괄호를 잘못치면 다 틀린다 ) 나중에 반복
print(y)
print(y.shape)   # ( 1459 , ) # 벡터가 1개 그래서 최종 아웃풋 갯수는 1개   ( 여기까지가 데이터 )

# test.csv = test_set.dropna(axis=0,how='any')



x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.7,
        shuffle=True,
        random_state=31)

# np.logical_or(x, y)
# print(x = train_set.info(x))
# print(train_set.dropna( subset['Age']))
# print(pd.isna('nan'))

#2. 모델구성
model = Sequential()
model.add(Dense(36, input_dim=9))          # 행 무시 열 우선 필수 
model.add(Dense(18))
model.add(Dense(18))  
model.add(Dense(1))

#3. 컴파일 훈련
start_time = time.time()

model.compile(loss='mse', optimizer = 'adam')        # 평가지표는 프레딕트 결과값 어쩌구 저쩌구 해서 mse 로 가능 비슷하면 된다 
model.fit(x_train, y_train, epochs=100, batch_size=200)

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
     return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)



y_summit = model.predict(test_set)

# print(y_summit)
# print(y_summit.shape) # (715, 1)

submission_set = pd.read_csv(path + 'submission.csv', # + 명령어는 문자를 앞문자와 더해줌
                             index_col=0) # index_col=n n번째 컬럼을 인덱스로 인식

print(submission_set)

submission_set['count'] = y_summit

# print(submission_set)


submission_set.to_csv('./_data/ddarung/submission.csv', index = True)

y_predict = model.predict(test_set)




# loss :  3325.189697265625
# RMSE :  57.66445714804163

# loss :  2915.238525390625
# RMSE :  53.99295020654411





# y_summit = model.predict(test_set)

# print(y_summit)
# print(y_summit.shape) # (715,1)

# submission_set['count'] = y_summit
# print(submission_set)
# submission_set.to_csv("C:\study\_data\ddarung", index=True)


# submission['count'] = y_summit        
# submission = submission.fillna(submission.mean())
# submission.to_csv(path + 'submission.csv', index=False)