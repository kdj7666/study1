#1. R2를 음수가 아닌 0.5 이하로 만들것  ( 강제적으로 나쁜 모델을 만들것 )
#2. 데이터 건들지 마
#3. 레이어는 인풋 아웃풋 포함 7개 이상
#4. batch_size = 1
#5. 히든레이어와 노드는 10개 이상 100개 이하 
#6. train 70% 
#7. epoch 100번 이상
#8. loss지표는 mse , mae 
# [ 실습 시작 ] 



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,8,9,3,8,12,13,8,17,12,16,19,11,12])

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.7, shuffle=False, random_state=66)

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=1))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일 , 훈련
model.compile(loss='mae', optimizer='adam')     # 회귀 모델의 대표적인 평가 지표 중에 하나 == R2(R제곱) R2수치가 높을수로 좋다 
model.fit(x_train, y_train, epochs=300, batch_size=1 )

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x)  

from sklearn.metrics import r2_score         # metrics 행렬 
r2 = r2_score(y,y_predict)
print('r2score : ', r2)

# loss :  53.707828521728516   훈련값 110 1회   layer 25개 
# r2score :  0.07638325995120587

# loss :  46.294219970703125    훈련값 110 2회   ㄹㅇㅇ 25개 
# r2score :  0.20200568748491798

# loss :  16.924110412597656    훈련값 300 1회   ㄹㅇㅇ 25개 
# r2score :  0.6279013377800586

# loss :  3.426213502883911      훈련값 300 2회   ㄹㅇㅇ 동일
# r2score :  0.6745503436837866

나머지는 집에서 좀더 찾고 리스트 공부 마저 다 쓰고 오늘한것 다시 적어보기 

# import matplotlib.pyplot as plt

# plt.scatter(x, y)    # 점을 흩뿌리다 ( 사전적 의미 )
# plt.plot(x, y_predict, color='red')  # 그려주어라 선을 색은 빨간색으로
# plt.show()          # 보여주라 점을 맵에 #평가지표는 항상 2개

