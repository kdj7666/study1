#1. 데이터
import numpy as np
from sklearn.metrics import log_loss
x = np.array([1,2,3,5,4])       
y = np.array([1,2,3,4,5])

#2. .모델구성 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential() 
model.add(Dense(4, input_dim=1))         
model.add(Dense(5))  # 오차범위가 훈련양으로 안줄어들면 히든레이어를 늘려도되고 node(뉴런)의 갯수를 늘려도 된다
model.add(Dense(3))  # 하이퍼 파라미터 튜닝 
model.add(Dense(2))  
model.add(Dense(1)) 

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')         
model.fit(x, y, epochs=880, batch_size=1) #가중치 보관 밑 xy로 # 훈련양 = 숫자
# x y 훈련양을 통으로 달라 #최소의 로스 최적의 웨이트


#4. 평가, 예측
loss = model.evaluate(x, y) # x y 값을 평가할것이다 그 값을 로스에 넣어주세요 # 웨이트값 # 로스값 
print("loss : ", loss) #y=wx+b x값을 범위 밖 수를 넣을때 y는 예측값 #최종로스값을 여기다 넣어주세요

result = model.predict([113]) 
print('113의 예측값은 : ', result) # 항상 결과값은 밑에 주석으로 첨부할것

