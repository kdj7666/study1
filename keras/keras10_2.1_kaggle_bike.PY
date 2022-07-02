# 캐글 바이크 문제풀이 

# 데이콘 따릉이 문제풀이 
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import nan_euclidean_distances, r2_score, mean_squared_error

#1. data  # 10번 경로  +는 문자가 연결이 된다
path = './_data2/bike/' 

train_set = pd.read_csv(path + 'train.csv', 
                        index_col=0)
print(train_set)             
print(train_set.shape)   # 10886 , 11

test_set = pd.read_csv(path + 'test.csv',
                       index_col=0)   

submission = pd.read_csv(path + 'sampleSubmission.csv')

def relu(a):
    return np.maximum(0,a)

print(test_set)
print(test_set.shape)   # 6493 , 8

print(train_set.columns) 
print(train_set.info())       
print(train_set.describe())  

x = train_set.drop(['count','casual','registered',], axis=1)
print(x)
print(x.columns)
print(x.shape)  # 10886 , 10 

y = train_set['count'] 
print(y)
print(y.shape)  # 10886 , 

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.949,
        shuffle=True,
        random_state=100)

# np.logical_or(x, y)
# print(x = train_set.info(x))
# print(train_set.dropna( subset['Age']))
# print(pd.isna('nan'))

#2. 모델구성
model = Sequential()
model.add(Dense(40, input_dim=8))          # 행 무시 열 우선 필수 
model.add(Dense(80, activation='swish'))
model.add(Dense(40, activation='swish'))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss='mae', optimizer = 'adam')        # 평가지표는 프레딕트 결과값 어쩌구 저쩌구 해서 mse 로 가능 비슷하면 된다 
model.fit(x_train, y_train, epochs=2450, batch_size=6600) 

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
     return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

y_summit = model.predict(test_set)
y_summit2 = relu(y_summit)
# print(y_summit)
# print(y_summit.shape) # (715,1)

######################## .to_csv()를 사용해서 아이디값 안됨 카운트값 순서대로 
### sampleSubmission.csv를 완성하시오 !!! ( 과제 겸 실습 )
# dataframe = pd.DataFrame(y_summit)
# dataframe.to_csv('.csv')

submission['count'] = y_summit2 
# submission = submission.fillna(submission.mean())
submission.to_csv(path + 'sampleSubmission.csv', index=False)


# loss :  95.6976089477539 # 0.949 True 100 
# RMSE :  134.5765159476751 1400 6600  40 80 80 80 80 40 1

# loss :  94.6959228515625
# RMSE :  136.43502128786216

