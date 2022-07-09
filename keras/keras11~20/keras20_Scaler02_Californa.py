from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense 
from gc import callbacks
import numpy as np 
import time
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

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

# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from sklearn.preprocessing import MaxAbsScaler, RobustScaler
scaler = RobustScaler()
# scaler = MaxAbsScaler()

# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_train))   # 0.0
print(np.max(x_train))   # 0.0 컬럼별로 나누어주어야 한다
print(np.min(x_test))
print(np.max(x_test))



#2. 모델구성
model = Sequential()
model.add(Dense(32, input_dim=8))
model.add(Dense(50, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(40))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일 , 훈련
start_time = time.time()

model.compile(loss='mse', optimizer='adam')     # 회귀 모델의 대표적인 평가 지표 중에 하나 == R2(R제곱) R2수치가 높을수로 좋다 
# from tensorflow.keras.callbacks import EarlyStopping
earlystopping = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1,
                              restore_best_weights=True)

a = model.fit(x_train, y_train, epochs=300, batch_size=100,
          validation_split=0.2,
          callbacks=[earlystopping],
          verbose=1)

print(a)
print(a.history['val_loss'])

end_time = time.time()-start_time
#4. 평가, 예측

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test) 
from sklearn.metrics import r2_score         # metrics 행렬 
r2 = r2_score(y_test, y_predict)
print('r2score : ', r2)
print('걸린시간 : ', end_time)

# 없음 
# loss :  0.5199254155158997
# r2score :  0.5992703644147576
# 걸린시간 :  75.85854959487915

# min max 
# loss :  0.31428590416908264
# r2score :  0.7577658751366537
# 걸린시간 :  41.788813829422

# standar 
# loss :  0.3052352964878082
# r2score :  0.764741548014421
# 걸린시간 :  41.15874886512756


# RobustScaler
# loss :  0.33525365591049194
# r2score :  0.741605030861392
# 걸린시간 :  20.488325595855713

# MaxAbsScaler
# loss :  0.3065764605998993
# r2score :  0.7637078840710475
# 걸린시간 :  146.54955315589905
