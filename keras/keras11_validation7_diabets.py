from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 

datasets = load_diabetes()       # 집합의 이름은 상관없다 다른걸로 해도 
x = datasets.data
y = datasets.target

print(x)                      # 데이터값 전처리데이터 
print(y)                      # 당뇨 수치 ( 전처리 데이터 안됨 비교값이기 때문에 전처리데이터가 필요없음 )

print(x.shape, y.shape)       #  (442, 10 )   (442)

print(datasets.feature_names)

# [ 실습 ]
# R2 0.62이상

print(datasets.DESCR)         #피쳐 아주중요 따로 찾아볼것 
x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.84, shuffle=False,) #random_state=100)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=10))
model.add(Dense(30))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(50))
model.add(Dense(1))

#3. 컴파일 , 훈련
<<<<<<< HEAD
model.compile(loss='mse', optimizer='adam')     # 회귀 모델의 대표적인 평가 지표 중에 하나 == R2(R제곱) R2수치가 높을수로 좋다 
=======
model.compile(loss='mae', optimizer='adam')     # 회귀 모델의 대표적인 평가 지표 중에 하나 == R2(R제곱) R2수치가 높을수로 좋다 
>>>>>>> 5f0219be1a123677777cf3b5268873b73cb5b912
model.fit(x_train, y_train, epochs=210, batch_size=10,
          validation_split=0.45)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)  

from sklearn.metrics import r2_score         # metrics 행렬 
r2 = r2_score(y_test, y_predict)
print('r2score : ', r2)

# -----------------------------------------------------------
# 검증 전 

# loss :  33.37435531616211   1회  0.9 False [10 20 30 20 10 1]
# r2score :  0.688494931960127     150 5 

# loss :  32.625125885009766 동일 2회 
# r2score :  0.6994415471161121

# loss :  33.589744567871094 동일 3회 
# r2score :  0.6843383946389526

# ------------------------------------------------------------
# 검증 후 

# loss :  33.36128234863281 1회 동일 
# r2score :  0.6747233317498502

# loss :  38.329933166503906 2회 동일 
# r2score :  0.5807784221550469

# loss :  33.784568786621094 3회 동일 
# r2score :  0.6648666399848596

# ------------------------------------------------

# 검증 후 튜닝 변경 

# loss :  39.26826095581055  1회 0.84 False [10 30 40 50 40 60 1]
# r2score :  0.5591262634166727  210 10 0.45

# loss :  38.830413818359375  2회 동일 
# r2score :  0.5582336660186611 

# loss :  39.38062286376953 3회 동일 
# r2score :  0.5508993737186549

