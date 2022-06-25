# keras08_1_boston.py 복붙 
from tabnanny import verbose
from sklearn. datasets import load_boston        # 사이킷 런에는 예제 문제가 있고 데이터가 있다
import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 
from sklearn.model_selection import train_test_split

# 1. 데이터

datasets = load_boston()              # load_boston 에서 x, y  데이터를 추출 한다 
x = datasets.data          #  ( x 의 데이터 로 )
y = datasets.target        #  ( y 의 값을 구한다 )

print(x)     
print(y)     

print(x.shape, y.shape)     # x = (506, 13)  y [506개의 스칼라에 1 벡터] = (506,)  506개의 데이터 13개의 컬럼 인풋 13 아웃풋 1

print(datasets.feature_names)       # ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'        
                                    # 'B' 'LSTAT']  범죄 세금 'b' 흑인  [ 조심 ] 오류 찾아볼것 

print(datasets.DESCR)         #피쳐 아주중요 따로 찾아볼것 
x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.7, shuffle=True, random_state=46)

# 2. 모델구성 

model = Sequential()
model.add(Dense(26, input_dim=13))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

import time
#3. 컴파일 , 훈련
model.compile(loss='mse', optimizer='adam')     # 회귀 모델의 대표적인 평가 지표 중에 하나 == R2(R제곱) R2수치가 높을수로 좋다 

start_time = time.time()        # 현재시간을 알려준다 몇시 몇분 몇초 까지 
print(start_time)       # 1656032971.0989587  ->  # 1656033205.2840586 이만큼의 시간이 지나갔다 ( 훈련 시간이 얼마나 경과했는지 보여줄수있다 [ 1번 2번 차이 ] )
model.fit(x_train, y_train, epochs=50, batch_size=1, verbose=0)          # verbose = 장황한 , 말이많은  ( 훈련과정을 안보여준다 ) [ 성능차이없다 메모리 열 등 ]
                                                                         # 훈련과정이 없으면 시간이 줄어든다 0.01초 보다 빨리 연산하면 느려진다 
end_time = time.time() - start_time

print("걸린시간 : ","초", end_time)

'''
verbose 0 일때 : 걸린시간 :  10.214722394943237     / 출력없다.

verbose 1 일때 : 걸린시간 :  12.384771823883057 걸린시가니 2초    / 잔소리많다

verbose 2 일때 : 걸린시간 :  10.611039161682129     / 프로그래스바 없다. 

verbose 3 일때 : 걸린시간 :  10.450383186340332    / epoch만 나온다 

verbose 4 일때 : 걸린시간 :  10.475086450576782    / epoch만 나온다 

verbose 5 일때 : 걸린시간 :  10.156696081161499     / 동일하다 


'''
