import numpy as np 
from tensorflow.keras.models import Sequential      
from tensorflow.keras.layers import Dense     

#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],                        # y=wx + b   w=1, b=10
             [1, 1, 1, 1, 2, 1.3, 1.4, 1.5, 1.6, 1.4],
             [9,8,7,6,5,4,3,2,1,0]]     # 열의 갯수는 반드시 맞춘다 
             )                                                  #  (2,10) 을 (10,2)
y = np.array([11,12,13,14,15,16,17,18,19,20])
print(x.shape) # (2,10)
print(y.shape) # (10,) -> (10,1) 이라 할수 있다

# x = x.T            행열 변환 ( 14 15 16 )
# x = x.transpose()
# x = x.reshape(10,2)      <- 이것이 input_dim=2 특성  
x = x.T
print(x)
print(x.shape) # (10,2)

# ( 숙제 ) 모델을 완성하시오 
# 예측 : [[10 , 1.4 , 0]]

#2. 모델구성
model = Sequential()
model.add(Dense (5, input_dim=3)) #  <- 특성이 2개   열 피쳐 컬럼 특성 mldel.add(Dense (5, input_dim=1)) <- 특성이 1개 
model.add(Dense (4))
model.add(Dense (3))
model.add(Dense (2))
model.add(Dense (2))
model.add(Dense (1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=30, batch_size=1)                  # batch_size=3일때 123, 456, 789, 10 으로 훈련시킨다


#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss :', loss)
result = model.predict([[10, 1.4 ,0]]) # 열 특성 피쳐 컬럼 열이 같아야 한다 input_dim 동일
print('[10, 1.4, 0]의 예측값 : ', result)

# loss : 5.553555638471153e-06 훈련량 600 1회 
# [10, 1.4, 0]의 예측값 :  [[19.997051]]

# loss : 3.254218370329909e-07 훈련량 600 2회 
# [10, 1.4, 0]의 예측값 :  [[20.000502]]


