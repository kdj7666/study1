from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split # 앞에 소문자는 변수 또는 함수 함수로 인식
import numpy as np
from sqlalchemy import Time
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout, Conv2D, Flatten, LSTM
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler # 카멜케이스 문제는없다
from sklearn.preprocessing import MaxAbsScaler, RobustScaler # ( 찾아 본 후 정리 하고 keras20 파일 전부 적용 ) 둘중 이상치가 잘 골라지는게 있음 
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

# data
datasets = fetch_california_housing()
x = datasets.data
y = datasets['target']


print(x.shape) # (14447, 8) (6193, 8)
print(y.shape) # (14447,) (6193,)

x = x.reshape(20640,8,1)
print(x)

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.7,
        shuffle=True,
        random_state=33)


#================================================================= scaler 자료
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from sklearn.preprocessing import MaxAbsScaler, RobustScaler
# scaler = RobustScaler()
# scaler = MaxAbsScaler()

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# print(np.min(x_train))   # 0.0
# print(np.max(x_train))   # 0.0 컬럼별로 나누어주어야 한다
# print(np.min(x_test))
# print(np.max(x_test))

#=================================================================

# print(np.unique(y_train, return_counts=True))
# print(np.unique(y_test, return_counts=True))

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)


# scaler = RobustScaler()
# scaler = MaxAbsScaler()
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# print(np.min(x_train))   # 0.0
# print(np.max(x_train))   # 0.0 컬럼별로 나누어주어야 한다
# print(np.min(x_test))
# print(np.max(x_test))

# 2. 모델구성

model = Sequential()
model.add(LSTM(units=64, return_sequences=True,
               input_shape=(8,1)))
model.add(LSTM(32, return_sequences=False,
               activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1, activation='linear'))
# input1 = Input(shape=(13,))
# dense1 = Dense(10)(input1)
# dense2 = Dense(5, activation='relu')(dense1)
# dense3 = Dense(3, activation='relu')(dense2)
# output1 = Dense(1)(dense3)
# model = Model(inputs=input1, outputs=output1)

#3. 컴파일 , 훈련

start_time = time.time()

model.compile(loss='mae', optimizer='adam',
              metrics = ['mse'])    

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

earlystopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1,  
              restore_best_weights=True)

a = model.fit(x_train, y_train, epochs=10, batch_size=100,
          validation_split=0.2,
          callbacks = [earlystopping],
          verbose=1)              

print(a)
print(a.history['val_loss']) 

end_time = time.time()-start_time

#4. 평가, 예측

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)


# result = model.predict(y_pred)
y_predict = model.predict(x_test)  # 이 값이 54번 으로 

from sklearn.metrics import r2_score         # metrics 행렬 

r2 = r2_score(y_test, y_predict)
print('r2score : ', r2)
print('걸린시간 :', end_time)
# print(' 걸린 결과값 : ', result)


# loss :  [0.5595259070396423, 0.6302388906478882]
# r2score :  0.5302315172213028
# 걸린시간 : 158.77363324165344

