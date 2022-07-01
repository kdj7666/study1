from matplotlib import font_manager
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

print(datasets.feature_names)       # ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'        
                                    # 'B' 'LSTAT']  범죄 세금 'b' 흑인  [ 조심 ] 오류 찾아볼것 

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
hist = model.fit(x_train, y_train, epochs=100, batch_size=25,
          validation_split=0.25)   # a 대신에 hist 라고 쓰임 
print(hist)
print(hist.history['loss']) # 대괄호로 loss , val loss 값 출력 가능

import matplotlib.pyplot as plt

from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name() # 49 50 51 세줄은 한글때문에 필요한것 * 한국인 필수
rc('font', family=font_name)

plt.figure(figsize=(15,10)) # plt.show 의 칸 가로의 길이 세로의 길이 
plt.plot(hist.history['loss'], marker='.', c='red', label='loss') # 곡선의 꺾임 marker . c = 색깔 red label loss 

#그림그릴거야 # 데이터 값의 시각화를 해달라 

plt.plot(hist.history['val_loss'], marker=',', c='blue', label= 'val_loss')
plt.grid()  # plt.show 의 그래프에 눈금을 그린다 
plt.title('abcd') # 그래프 위의 제목 타이틀 
plt.ylabel('loss')    # 55번과 57번 색과 그래프의 선이 다름 표현해달라 
plt.xlabel('epochs')  # x는 epochs 수치를 표현해달라 
plt.legend(loc='upper right') # 라벨값 위치 생략시 빈자리에 생성
plt.show() # 이 그래프를 보여달라 




'''
#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)  # 이 값이 54번 으로 

from sklearn.metrics import r2_score         # metrics 행렬 
r2 = r2_score(y_test, y_predict)
print('r2score : ', r2)





'''



# <keras.callbacks.History object at 0x0000015F32465700> 케라스 파일 콜백에 있는 히스토리의 / 메모리 저장된 메모리 주소 



