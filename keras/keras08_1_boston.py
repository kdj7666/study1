from sklearn. datasets import load_boston        # 사이킷 런에는 예제 문제가 있고 데이터가 있다
import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 
from sklearn.model_selection import train_test_split
import time
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

print(datasets.feature_names)       # ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'        
                                    # 'B' 'LSTAT']  범죄 세금 'b' 흑인  [ 조심 ] 오류 찾아볼것 

print(datasets.DESCR)         #피쳐 아주중요 따로 찾아볼것 
x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.7, shuffle=True, random_state=31)

     # [ 실습 ] 아래를 완성할것
     # 1.  train 0.7
     # 2.  r2 0.8이상

# 2. 모델구성 

model = Sequential()
model.add(Dense(26, input_dim=13))
model.add(Dense(20, activation='swish'))
model.add(Dense(20, activation='swish'))
model.add(Dense(10, activation='swish'))
model.add(Dense(1))

#3. 컴파일 , 훈련
model.compile(loss='mse', optimizer='adam')     # 회귀 모델의 대표적인 평가 지표 중에 하나 == R2(R제곱) R2수치가 높을수로 좋다 
model.fit(x_train, y_train, epochs=30000, batch_size=5)

start_time = time.time()

end_time = time.time()
#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

print('걸린시간 : ', end_time)

y_predict = model.predict(x_test)  

from sklearn.metrics import r2_score         # metrics 행렬 
r2 = r2_score(y_test, y_predict)
print('r2score : ', r2)


# loss :  20.77730941772461         # 300 b 4   1
# r2score :  0.74851080871396

# loss :  19.995521545410156         # 300 b 4   2
# r2score :  0.7579735630745217

# loss :  18.821191787719727          # 300 4   3 
# r2score :  0.7721877249524539

# loss :  18.340871810913086        # 400 4 4 
# r2score :  0.7780014802579737

# loss :  17.845577239990234     ㄹㅇㅇ 26 20 10 1 ㅎㄹㄹ 600  b 40 true 
# r2score :  0.7839965841474423

# loss :  16.26223373413086       ㄹㅇㅇ 5 swish ㅎㄹㄹ 1000 b 50
# r2score :  0.8022913341280221   0.7 True 50 

# loss :  15.887526512145996
# r2score :  0.8068468499807665
