from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,4,7,6,9,8,9,11,10,13,17,10,11,16,17,11,12,20])

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.7, 
        shuffle=True,
        random_state=66)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))



#3. 컴파일 , 훈련
model.compile(loss='mse', optimizer='adam')     # 회귀 모델의 대표적인 평가 지표 중에 하나 == R2(R제곱) R2수치가 높을수로 좋다 
model.fit(x_train, y_train, epochs=400, batch_size=1 )

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x)
from sklearn.metrics import r2_score     # metrics 행렬 
r2 = r2_score(y, y_predict)              # R2 라는 지표는 
print('r2score : ', r2)                  # loss 값이 높고 R2 지표가 낮으면 좋다 (비례)


# import matplotlib.pyplot as plt

# plt.scatter(x, y)    # 점을 흩뿌리다 ( 사전적 의미 )
# plt.plot(x, y_predict, color='red')  # 그려주어라 선을 색은 빨간색으로
# plt.show()          # 보여주라 점을 맵에 #평가지표는 항상 2개

# loss 11 - 12 

# loss :  1.6520277261734009      훈련양 500번  layer 15 
# r2score :  0.8120837777054872 

# loss :  1.6974563598632812   훈련양 500번 ㄹㅇㅇ 동일 
# r2score :  0.8116337183523681

# 데이터가 많이 틀렷을때 
# loss :  3.612239122390747       훈련양 400 layer  12개  1회  
# r2score :  0.7515020100277605

# loss :  4.055657863616943   훈련양 ㄹㅇㅇ 동일  2회
# r2score :  0.7389339082785943

# loss :  3.3568837642669678   훈련양 ㄹㅇㅇ 동일 3회
# r2score :  0.7551037463930119





