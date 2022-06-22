from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,3,4,7,6,9,8,9,10,11,13,17,14,15,16,17,11,12,20])

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.7, 
        shuffle=True,
        random_state=66)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(70))
model.add(Dense(50))
model.add(Dense(45))
model.add(Dense(30))
model.add(Dense(67))
model.add(Dense(46))
model.add(Dense(38))
model.add(Dense(48))
model.add(Dense(67))
model.add(Dense(45))
model.add(Dense(44))
model.add(Dense(97))
model.add(Dense(68))
model.add(Dense(77))
model.add(Dense(1))



#3. 컴파일 , 훈련
model.compile(loss='mse', optimizer='adam')     # 회귀 모델의 대표적인 평가 지표 중에 하나 == R2(R제곱) R2수치가 높을수로 좋다 
model.fit(x_train, y_train, epochs=500, batch_size=1 )

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x)
from sklearn.metrics import r2_score     # metrics 행렬 
r2 = r2_score(y, y_predict)
print('r2score : ', r2)


# import matplotlib.pyplot as plt

# plt.scatter(x, y)    # 점을 흩뿌리다 ( 사전적 의미 )
# plt.plot(x, y_predict, color='red')  # 그려주어라 선을 색은 빨간색으로
# plt.show()          # 보여주라 점을 맵에 #평가지표는 항상 2개

# loss 11 - 12 


