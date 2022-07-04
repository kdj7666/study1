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
model.fit(x_train, y_train, epochs=1500, batch_size=400,
          validation_split=0.35)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test) 

from sklearn.metrics import r2_score         # metrics 행렬 
r2 = r2_score(y_test, y_predict)
print('r2score : ', r2)

# 검증 전 
# loss :  0.5835319757461548  True 0.9 31  16 34 24 20 12 1 
# r2score :  0.5636667427506528   1000 300
 
# loss :  0.5923864841461182  동일 2회
# r2score :  0.55704577233998

# loss :  0.5876272916793823 동일 3회
# r2score :  0.560604462382521

# loss :  0.5828465819358826 동일 4회 
# r2score :  0.5641792756410814

#------------------------------------------

# 검증 후 

# loss :  0.6352787017822266 동일 1회
# r2score :  0.5249733493717579

# loss :  0.5955377817153931 동일 2회 
# r2score :  0.5546894334252346

# loss :  0.5969669818878174 동일 3회 
# r2score :  0.5536207281380396

# 검증 후 튜닝 값 변경

# loss :  0.636787474155426  1회    0.9 True 77 32 50 60 40 10 1 
# r2score :  0.5091995939672236     1500 400 0.35

# loss :  0.652357280254364 2회 동일 
# r2score :  0.49719915458148023

# loss :  0.677389919757843 3회 동일 
# r2score :  0.4779054470042128

