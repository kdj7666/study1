import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. date 
x = np.array([range(10)])   # range 라는건 범위 안 숫자
# print(range(10))          #  파이썬의 숫자는 0부터 
# for i in range(10):      # 파이썬 for = 반복문 
#     print(i)           #  i라는 이수에 대입 10번을 대입해라
print(x.shape) # (3,10)
                                   # 14번 라인까지 실행은 빨강으로 고정 후 그냥 f5 1,10 , 10,1
x = np.transpose(x)
print(x.shape) # (10,3)
                                 # x 는 10,3 y는 10.2 y를 아웃풋 2회
y = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
             [9,8,7,6,5,4,3,2,1,0]])
y = np.transpose(y)
print(y.shape)        # (2,10) -> (10,2)

 # 2. model
 #[ 실습 ] # 예측 : [[9, 30, 210]] 예상 10 , 1.9  로스가 얼마나 주느냐
 
model = Sequential()
model.add(Dense (5, input_dim=1)) #  <- 특성이 2개   열 피쳐 컬럼 특성 mldel.add(Dense (5, input_dim=1)) <- 특성이 1개 
model.add(Dense (6))
model.add(Dense (5))
model.add(Dense (4))
model.add(Dense (5))
model.add(Dense (4))
model.add(Dense (3))
model.add(Dense (3))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1450)           


#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss :', loss)                                   # 인공지능 훈련시 
result = model.predict([9]) #PREDICT ( 예측 )           # 평가 데이터와 학습 데이터는 달라야 한다. 
print('[9]의 예측값 : ', result) # 10, 1.9 ,0           # 훈련용 데이터와 평가용 데이터는 달라야한다.  


