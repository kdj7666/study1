from sklearn. datasets import load_boston        # 사이킷 런에는 예제 문제가 있고 데이터가 있다
import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 
from sklearn.model_selection import train_test_split

# 1. 데이터

datasets = load_boston()              # load_boston 에서 x, y  데이터를 추출 한다 
x = datasets.data          #  ( x 의 데이터 로 )
y = datasets.target        #  ( y 의 값을 구한다 )

print(x)     # ( 보스턴 집 값을 위한 데이터 ) 

             # 연산자 찾아보기 ( 4.7410e-02)  e-ㅁㅁ 는 소숫점 뒤 0의 갯수   
             #                ( 3.9690e+02 ) e+ㅁㅁ 는 소숫점 앞 0의 갯수 
             # 데이터 전처리 0 ~ 1 까지 0 50 100 을 0 0.5 1 로 바꿀때 y값은 동일하게 가리킨다 
print(y)     # ( 보스턴 집 값을 위한 데이터를 사용하여 나온 보스턴 집 값 )

print(x.shape, y.shape)     # x = (506, 13)  y [506개의 스칼라에 1 벡터] = (506,)  506개의 데이터 13개의 컬럼 인풋 13 아웃풋 1
<<<<<<< HEAD
                 # 22번 싸이킷런에만 있는 명령어 
print(datasets.feature_names)       # ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' # 'B' 'LSTAT']  범죄 세금 'b' 흑인  [ 조심 ] 오류 찾아볼것 
=======

print(datasets.feature_names)       # ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'        
                                    # 'B' 'LSTAT']  범죄 세금 'b' 흑인  [ 조심 ] 오류 찾아볼것 

>>>>>>> 5f0219be1a123677777cf3b5268873b73cb5b912
print(datasets.DESCR)         #피쳐 아주중요 따로 찾아볼것 
x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.7, shuffle=True, random_state=55)

# 2. 모델구성

model = Sequential()
model.add(Dense(26, input_dim=13))
model.add(Dense(40))
model.add(Dense(50))
model.add(Dense(60))
model.add(Dense(1))

#3. 컴파일 , 훈련
model.compile(loss='mse', optimizer='adam')     # 회귀 모델의 대표적인 평가 지표 중에 하나 == R2(R제곱) R2수치가 높을수로 좋다 
model.fit(x_train, y_train, epochs=820, batch_size=25,
          validation_split=0.25)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)  # 이 값이 54번 으로 

from sklearn.metrics import r2_score         # metrics 행렬 
r2 = r2_score(y_test, y_predict)
print('r2score : ', r2)

# 검증 전 결과값 

# loss :  16.504552841186523  0.7 true 50 
# r2score :  0.7993453422042345 26 20 20 10 1 e 1000 b 50 

# loss :  17.482885360717773    78번 동일 
# r2score :  0.7874512046527672

# loss :  16.888813018798828
# r2score :  0.7946736682017962 동일 

# loss :  16.26223373413086       ㄹㅇㅇ 5 swish ㅎㄹㄹ 1000 b 50
# r2score :  0.8022913341280221   0.7 True 50 

# ----------------------------------------------------------------------

# 검증 적용 후

# loss :  19.271127700805664   0.7 True 45 [ 26 20 20 10 1 ] 1회
# r2score :  0.800477715217355  e 500 b 30 val = 0.25

# loss :  28.190738677978516 동일 2회
# r2score :  0.708129168319559

# loss :  23.92193031311035  동일 3회
# r2score :  0.7523259655500283

# 검증 적용 후 튜닝 값 변경 

# loss :  27.269290924072266 0.7 True 55 [ 26 40 50 60 1 ] 1회
# r2score :  0.6233232773389685 e 820 b 25 val = 0.25

# loss :  20.74823760986328   동일 2회
# r2score :  0.7133999712507386

# loss :  19.21137809753418   동일 3회
# r2score :  0.7346289921906355

# loss :  20.784631729125977 동일 4회 
# r2score :  0.7128972676807577

