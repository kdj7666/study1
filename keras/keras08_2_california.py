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
<<<<<<< HEAD
        train_size=0.9, shuffle=True, random_state=35)
=======
        train_size=0.9, shuffle=True, random_state=31)
>>>>>>> 5f0219be1a123677777cf3b5268873b73cb5b912

#2. 모델구성
model = Sequential()
model.add(Dense(16, input_dim=8))
model.add(Dense(34))
model.add(Dense(24))
model.add(Dense(20))
model.add(Dense(12))
model.add(Dense(1))

#3. 컴파일 , 훈련
model.compile(loss='mse', optimizer='adam')     # 회귀 모델의 대표적인 평가 지표 중에 하나 == R2(R제곱) R2수치가 높을수로 좋다 
<<<<<<< HEAD
model.fit(x_train, y_train, epochs=25000, batch_size=300)
=======
model.fit(x_train, y_train, epochs=1000, batch_size=300)
>>>>>>> 5f0219be1a123677777cf3b5268873b73cb5b912

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)  

from sklearn.metrics import r2_score         # metrics 행렬 
r2 = r2_score(y_test, y_predict)
print('r2score : ', r2)


# loss :  0.6287996172904968
# r2score :  0.541748011203979

# loss :  0.6250039935112       # 훈련량 5000 배치 1000 레이어층 8 
# r2score :  0.5445141668253644

# loss :  0.6008335947990417  훈련량 5000 배치 1000 레이어층 2배
# r2score :  0.5621289239912706

# loss :  0.6219940185546875       훈련량 1000 배치 300 레이어층 
# model.add(Dense(16, input_dim=8))
# model.add(Dense(71))
# model.add(Dense(34))
# model.add(Dense(24))
# model.add(Dense(20))
# model.add(Dense(12))
# model.add(Dense(1))
# r2score :  0.5538272124002598


# loss :  0.6191340088844299        훈련량 1000 레이어층 71제거 배치 200
# r2score :  0.5558788847224494     트레이닝 사이즈 0.8 


#loss :  0.5983598232269287     ( 71 번 동일 )    랜덤 86
# r2score :  0.5745712183147    트레이닝 사이즈 0.9

# loss :  0.31384748220443726      훈련량 8000 레이어층 같음 배치 200 랜덤 86
# r2score :  0.6121719072616624    트레이닝 사이즈 0.9 

# loss :  0.315838098526001         동일 2회째 
# r2score :  0.6097119171502551

# loss :  0.5835319757461548  True 0.9 31  16 34 24 20 12 1 
# r2score :  0.5636667427506528   1000 300
 
# loss :  0.5923864841461182  동일 2회
# r2score :  0.55704577233998

# loss :  0.5876272916793823 동일 3회
# r2score :  0.560604462382521

# loss :  0.5828465819358826 동일 4회 
# r2score :  0.5641792756410814