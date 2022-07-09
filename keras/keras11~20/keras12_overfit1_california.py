from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 

from sklearn.datasets import fetch_california_housing
datasets = fetch_california_housing()   # 보스턴 값에서 이제는 fetch_california_housing() 으로 계산하라 
x = datasets.data  
y = datasets.target 
print(x)
print(y)
print(x.shape, y.shape)    # (20640, 8) (20640,)

print(datasets.feature_names) # - MedInc        median income in block group
                              # - HouseAge      median house age in block group
                              # - AveRooms      average number of rooms per household
                              # - AveBedrms     average number of bedrooms per household
                              # - Population    block group population
                              # - AveOccup      average number of household members
                              # - Latitude      block group latitude
                              # - Longitude     block group longitude

print(datasets.DESCR)         #피쳐 아주중요 따로 찾아볼것 
x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.9, shuffle=True, random_state=77)

#2. 모델구성
model = Sequential()
model.add(Dense(32, input_dim=8))
model.add(Dense(50))
model.add(Dense(60))
model.add(Dense(40))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일 , 훈련
model.compile(loss='mse', optimizer='adam')     # 회귀 모델의 대표적인 평가 지표 중에 하나 == R2(R제곱) R2수치가 높을수로 좋다 
late = model.fit(x_train, y_train, epochs=100, batch_size=400,
          validation_split=0.2)

print(late)
print(late.history['loss'])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test) 

from sklearn.metrics import r2_score         # metrics 행렬 
r2 = r2_score(y_test, y_predict)
print('r2score : ', r2)

import matplotlib.pyplot as plt

plt.figure(figsize=(9,6)) #칸만들기
plt.plot(late.history['loss'], marker='.', c='pink', label='loss') # 곡선의 꺾임 marker . c = 색깔 red label loss 

#그림그릴거야

plt.plot(late.history['val_loss'], marker=',', c='black', label= 'val_loss')
plt.grid()
plt.title('late') # 그래프 위의 제목 타이틀 
plt.ylabel('loss')    # 55번과 57번 색과 그래프의 선이 다름 표현해달라 
plt.xlabel('epochs')  # x는 epochs 수치를 표현해달라 
plt.legend(loc='upper right') # 라벨값 위치 생략시 빈자리에 생성
plt.show() # 이 그래프를 보여달라 

