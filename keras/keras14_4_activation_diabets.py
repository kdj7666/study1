from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 
from gc import callbacks
import numpy as np 
from sklearn.datasets import load_diabetes
datasets = load_diabetes() 
x = datasets.data  
y = datasets.target 
print(x)
print(y)
print(x.shape, y.shape) 

print(datasets.feature_names)

print(datasets.DESCR)         #피쳐 아주중요 따로 찾아볼것 
x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.9, shuffle=True, random_state=77)

#2. 모델구성
model = Sequential()
model.add(Dense(32, input_dim=10))
model.add(Dense(50, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(40))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일 , 훈련
model.compile(loss='mse', optimizer='adam')     # 회귀 모델의 대표적인 평가 지표 중에 하나 == R2(R제곱) R2수치가 높을수로 좋다 
from tensorflow.keras.callbacks import EarlyStopping
earlystopping = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1,
                              restore_best_weights=True)

a = model.fit(x_train, y_train, epochs=3000, batch_size=400,
          validation_split=0.2,
          callbacks=[earlystopping],
          verbose=1)

print(a)
print(a.history['val_loss'])

#4. 평가, 예측

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test) 
from sklearn.metrics import r2_score         # metrics 행렬 
r2 = r2_score(y_test, y_predict)
print('r2score : ', r2)

# 검증 전 

# loss :  32.625125885009766 
# r2score :  0.6994415471161121

# 검증 후 

# loss :  33.36128234863281  
# r2score :  0.6747233317498502

# 검증 후 튜닝 변경 

# loss :  39.26826095581055 
# r2score :  0.5591262634166727


# earlystopping 적용 후 

# loss :  3254.716552734375
# r2score :  0.5189185896849666

# activation 적용 후 

# loss :  2925.55078125
# r2score :  0.5675727683035585
