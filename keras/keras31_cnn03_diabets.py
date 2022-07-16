from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Conv2D, Flatten, Conv1D
from gc import callbacks
import numpy as np 
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
datasets = load_diabetes() 
x = datasets.data  
y = datasets.target 

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.7, shuffle=True, random_state=77)

print(x_train.shape, x_test.shape) # (309, 10) (133, 10)
print(y_train.shape, y_test.shape) # (309,) (133,)

x_train = x_train.reshape(309,5,2,1)
x_test = x_test.reshape(133,5,2,1)
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

print(np.unique(y_train, return_counts=True))
print(np.unique(y_test, return_counts=True))

#2. 모델구성

# model = Sequential()
# model.add(Dense(32, input_dim=10))
# model.add(Dense(50, activation='relu'))
# model.add(Dense(60, activation='relu'))
# model.add(Dense(40))
# model.add(Dense(10))
# model.add(Dense(1))

# input1 = Input(shape=(10,))
# dense1 = Dense(10)(input1)
# dense2 = Dense(5, activation='relu')(dense1)
# dense3 = Dense(3, activation='relu')(dense2)
# output1 = Dense(1)(dense3)
# model = Model(inputs=input1, outputs=output1)

model = Sequential()
model.add(Conv2D(filters=240, kernel_size=(2,1),
                 padding='same', input_shape=(5,2,1)))
model.add(Conv2D(120, (2,1),
                 padding='valid',
                 activation='relu'))
model.add(Conv2D(60, (2,1),
                 padding='valid',
                 activation='relu'))
model.add(Flatten())
model.add(Dense(30, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(1, activation='linear'))

#3. 컴파일 , 훈련

start_time = time.time()

model.compile(loss='mse', optimizer='adam',
              metrics=['mae'])     # 회귀 모델의 대표적인 평가 지표 중에 하나 == R2(R제곱) R2수치가 높을수로 좋다 

earlystopping = EarlyStopping(monitor='val_loss', patience=300, mode='min', verbose=1,
                              restore_best_weights=True)

a = model.fit(x_train, y_train, epochs=1550, batch_size=100,
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

r2score :  0.5009690974568957