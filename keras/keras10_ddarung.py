# 데이콘 따릉이 문제풀이 
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import nan_euclidean_distances, r2_score, mean_squared_error

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


x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.9, shuffle=True, random_state=86)

# np.logical_or(x, y)
# print(x = train_set.info(x))
# print(train_set.dropna( subset['Age']))
# print(pd.isna('nan'))

#2. 모델구성
model = Sequential()
model.add(Dense(72, input_dim=9))          # 행 무시 열 우선 필수 
model.add(Dense(72))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(200))
model.add(Dense(200))
model.add(Dense(200))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer = 'adam')        # 평가지표는 프레딕트 결과값 어쩌구 저쩌구 해서 mse 로 가능 비슷하면 된다 
model.fit(x_train, y_train, epochs=1500, batch_size=100, verbose=0)

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
     return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)
# loss :  2874.76953125  ㅎㄹㄹ 1000 ㅂㅊ 1500 
# loss :  2852.6865234375 ㅎㄹㄹ 800 ㅂㅊ 300 
# loss :  2840.17041015625 ㅎㄹㄹ 800 ㅂㅊ 300 ㄹㅇㅇ 10 

# def RMSE(y_test, y_predict):
#      return np.sqrt(mean_squared_error[y_test, y_predict])

# rmse = RMSE(y_test, y_predict)
# RMSE 값을 비교하겠다 y 테스트와 y프레딕트
# 숫자가 커지기 때문에 루트를 한번 씌운다 
# 루트를 씌우고 나서 나온 값 
# 원래의 데이터 값과 y_predict 의 예측값을 항상 비교한다
# np.sprt 로 루트를 씌웟다  
# 민 스퀏 에러에 와이 테스트와 와이 프레딕트 를 집어 넣었고 그것을 RMSE 로 출력해주겠다
# def 함수를 만들거야 RMSE( 와이 테스트와 와이 프레딕트란 값을 받아들일거야)
# 내가 받아들인 와이와 와이프레딕트 값을 사이킷런에 있는 민 스퀏 에러에 해라 거기에 루트를 씌우겟다 
# 결과값을 돌려주겠다 


# 결과값이 잘나오면 첨부할것 
# loss :  2298.373046875
# RMSE :  47.941351250276846

# loss :  2199.09912109375
# RMSE :  46.89455305190569






