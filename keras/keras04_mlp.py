import numpy as np 
from tensorflow.keras.models import Sequential      
from tensorflow.keras.layers import Dense     

#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],                        # y=wx + b   w=1, b=10
             [1, 1, 1, 1, 2, 1.3, 1.4, 1.5, 1.6, 1.4]]      # 열의 갯수는 반드시 맞춘다 
             )                                                 #  (2,10) 을 (10,2)
y = np.array([11,12,13,14,15,16,17,18,19,20])
print(x.shape) # (2,10)
print(y.shape) # (10,) -> (10,1) 이라 할수 있다

# x = x.T
# x = x.transpose()
# x = x.reshape(10,2)      <- 이것이 input_dim=2 특성  
x = x.T
print(x)
print(x.shape) # (10,2)



#2. 모델구성
model = Sequential()
model.add(Dense (5, input_dim=2)) #  <- 특성이 2개   열 피쳐 컬럼 특성 mldel.add(Dense (5, input_dim=1)) <- 특성이 1개 
model.add(Dense (4))
model.add(Dense (3))
model.add(Dense (2))
model.add(Dense (1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=400, batch_size=1)                  # batch_size=3일때 123, 456, 789, 10 으로 훈련시킨다


#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss :', loss)
result = model.predict([[10, 1.4]]) # 열 특성 피쳐 컬럼 열이 같아야 한다 input_dim 동일
print('[10, 1.4]의 예측값 : ', result)  # ValueError : Data cardinality is ambiguous:   값에 대한 에러가있다 : x사이즈는 2개 y사이즈는 10개다 = x값의 모양과 y값의 모양이 다르다
                                     # x sizes: 2 y sizes: 10
                                     
# loss : 0.7778943181037903 훈련량 400번 1회
# [10, 1.4]의 예측값 :  [[19.391918]]

# loss : 0.0524146631360054 훈련량 400번 2회
# [10, 1.4]의 예측값 :  [[19.843906]]


