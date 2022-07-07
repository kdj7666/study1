# https://dacon.io/competitions/open/235576/overview/description

# 데이콘 따릉이 문제풀이 
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import nan_euclidean_distances, r2_score, mean_squared_error
import time
#1. data  # 10번 경로  +는 문자가 연결이 된다
path = './_data/ddarung/' 
train_set = pd.read_csv(path + 'train.csv', 
                        index_col=0)         #index_col 1번째는 id 행의 이름이기때문에 계산 ㄴ
print(train_set)             
print(train_set.shape)       # 1459개의 열과 10개의 컬럼  (1459,10)

test_set = pd.read_csv(path + 'test.csv',
                       index_col=0)     #  예측에서 프레딕트로 쓸것이다

submission = pd.read_csv(path + 'submission.csv')

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
x = train_set.drop(['count'], axis=1)   # drop 날리다 카운트라는 줄을 날릴것이다 소숫점이 1개 
print(x)
print(x.columns)
print(x.shape) #  ( 1459 , 9 )

y = train_set['count']  # 이렇게 하면 빠진다 지금은 이정도 ( [ ] 대괄호를 잘못치면 다 틀린다 ) 나중에 반복
print(y)
print(y.shape)   # ( 1459 , ) # 벡터가 1개 그래서 최종 아웃풋 갯수는 1개   ( 여기까지가 데이터 )

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.989,
        shuffle=True,
        random_state=40)

# np.logical_or(x, y)
# print(x = train_set.info(x))
# print(train_set.dropna( subset['Age']))
# print(pd.isna('nan'))

#2. 모델구성
model = Sequential()
model.add(Dense(90, input_dim=9))          # 행 무시 열 우선 필수 
model.add(Dense(80, activation='swish'))
model.add(Dense(80, activation='swish'))  
model.add(Dense(50, activation='swish'))
model.add(Dense(50, activation='swish'))  
model.add(Dense(50, activation='swish'))   
model.add(Dense(1))

#3. 컴파일 훈련
start_time = time.time()
model.compile(loss='mae', optimizer = 'adam')        # 평가지표는 프레딕트 결과값 어쩌구 저쩌구 해서 mse 로 가능 비슷하면 된다 
model.fit(x_train, y_train, epochs=1000, batch_size=100) 
end_time = time.time()-start_time
#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
     return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

print('걸린시간 : ', end_time)

y_summit = model.predict(test_set)

# print(y_summit)
# print(y_summit.shape) # (715,1)

######################## .to_csv()를 사용해서 아이디값 안됨 카운트값 순서대로 
### submission.csv를 완성하시오 !!! ( 과제 겸 실습 )
# dataframe = pd.DataFrame(y_summit)
# dataframe.to_csv('.csv')

submission['count'] = y_summit        
submission = submission.fillna(submission.mean())
submission.to_csv(path + 'submission.csv', index=False)

# loss :  17.521583557128906 동일 
# RMSE :  21.03790096341299

# loss :  23.78106689453125
# RMSE :  34.51621760194604

# loss :  19.65888786315918
# RMSE :  34.915773545704084

# loss :  545.470947265625     동일 
# RMSE :  23.355318360896916

# loss :  398.7279968261719
# RMSE :  19.96817490940722
# epochs = 800
# batch_size  = 60
# teain_size = 0.989
# shuffle = True
# random_state = 100 
# layer 5층
# node 90 100 100 50 1   input 9 
# loss ' mse ' optimizer = 'adam' 

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


# loss :  2178.99462890625
# RMSE :  46.67970024480401

# loss :  1920.60986328125  ㅎㄹㄹ 1000 ㅂㅊㅅㅇㅈ 400 ㄹㅇㅇ 5개 함수 활성화
# RMSE :  43.82476213596008   트루로 섞고 0.9트레인사이즈 랜덤스테이트31

# loss :  1237.508056640625  ㅎㄹㄹ 2000 ㅂㅊㅅㅇㅈ 300 [ activation='swish' ] 함수 찾아볼것
# RMSE :  35.17823108072968 트루로 섞고 0.9 트레인사이즈 랜덤스테이트 31

# loss :  1358.9656982421875 동일 ㅎㄹㄹ 3000 ㅂㅊ 50 
# RMSE :  36.864151575638004

# loss :  660.1187744140625 ㅎㄹㄹ 600 ㅂㅊㅅㅇㅈ 60 activation='swish' 
# RMSE :  25.692777102099445 트루로 섞고 .989 트레인사이즈 랜덤스테이트 100

########################################################
# loss :  1173.5157470703125  ㅎㄹㄹ 500 ㅂㅊ 60 activation='swish' 2회 
# RMSE :  34.25661683053811 트루로 섞고 0.989 랜덤 스테이트 100 

# loss :  1147.8006591796875  동일 
# RMSE :  33.87920396789233

# loss :  956.2131958007812 동일 4회 
# RMSE :  30.922699751380193

# loss :  871.3640747070312 훈련량 600으로증가 동일 
# RMSE :  29.518876324167785

# loss :  813.6890258789062 훈련량 700으로 증가 동일 
# RMSE :  28.525237451873004

# loss :  795.462890625 훈련량 700 동일 4회 
# RMSE :  28.20395442089948

# loss :  708.4660034179688 훈련량 800 동일 2회 
# RMSE :  26.61702357657338

# loss :  42.667667388916016
# RMSE :  66.38048637184875
# 걸린시간 :  74.39030742645264
