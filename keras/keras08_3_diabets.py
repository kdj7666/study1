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
        train_size=0.9, shuffle=False,) #random_state=100)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=10))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일 , 훈련
model.compile(loss='mae', optimizer='adam')     # 회귀 모델의 대표적인 평가 지표 중에 하나 == R2(R제곱) R2수치가 높을수로 좋다 
model.fit(x_train, y_train, epochs=150, batch_size=5)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)  

from sklearn.metrics import r2_score         # metrics 행렬 
r2 = r2_score(y_test, y_predict)
print('r2score : ', r2)

# loss :  2649.696533203125      # 500 b 10 fdd 10 
# r2score :  0.5013769471116828

# loss :  33.49235534667969  # 훈련량 150 배치 10 ㄹㅇㅇ 6 개 loss ' mae ' 1회 
# r2score :  0.6909179158191338

# loss :  34.52722930908203     동일 
# r2score :  0.6680539770049845
