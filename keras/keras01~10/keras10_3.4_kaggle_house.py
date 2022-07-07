
# kaggle house 문제풀이 
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import nan_euclidean_distances, r2_score, mean_squared_error

#1. data  # 10번 경로  +는 문자가 연결이 된다
path = './_data3/house/' 
train_set = pd.read_csv(path + 'train.csv', 
                        index_col=0) 
print(train_set)             
print(train_set.shape)  # 1459 , 79

test_set = pd.read_csv(path + 'test.csv',
                       index_col=0)     

submission = pd.read_csv(path + 'sample_submission.csv')

print(test_set)
print(test_set.shape)  

print(train_set.columns) 
print(train_set.info())    
print(train_set.describe()) 

# #### 결측치 처리 1. 제거####
print(train_set.isnull().sum())
train_set = train_set.dropna()
print(train_set.isnull().sum())
print(train_set.shape)

#############################
x = train_set.drop(['SalePrice'], axis=1)   # drop 날리다 카운트라는 줄을 날릴것이다 소숫점이 1개 
print(x)
print(x.columns)
print(x.shape) #  ( 1459 , 9 )

y = train_set['SalePrice']  # 이렇게 하면 빠진다 지금은 이정도 ( [ ] 대괄호를 잘못치면 다 틀린다 ) 나중에 반복
print(y)
print(y.shape)   # ( 1459 , ) # 벡터가 1개 그래서 최종 아웃풋 갯수는 1개   ( 여기까지가 데이터 )

from sklearn.preprocessing import LabelEncoder
# encoder = LabelEncoder()
# encoder.fit(x)
# digit_label  = encoder.transform(pd.read_csv(path + 'train.csv'))
# print('encoder', encoder)
# print('encoder 결과',digit_label)

items=pd.read_csv(path + 'train.csv')
encoder = LabelEncoder()
encoder.fit(train_set)

labels = encoder.transform(items)
print('인코딩 변환값:',labels)

















x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.7,
        shuffle=True,
        random_state=50)

# np.logical_or(x, y)
# print(x = train_set.info(x))
# print(train_set.dropna( subset['Age']))
# print(pd.isna('nan'))

#2. 모델구성
model = Sequential()
model.add(Dense(90, input_dim=79))          # 행 무시 열 우선 필수 
model.add(Dense(80, activation='swish'))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer = 'adam')        # 평가지표는 프레딕트 결과값 어쩌구 저쩌구 해서 mse 로 가능 비슷하면 된다 
model.fit(x_train, y_train, epochs=100, batch_size=100) 

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
     return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

y_summit = model.predict(test_set)

print(y_summit)
print(y_summit.shape)

# submission['count'] = y_summit        
# submission = submission.fillna(submission.mean())
# submission.to_csv(path + 'submission.csv', index=False)


