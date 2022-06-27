from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense

datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x)
print(y)

print(x.shape, y.shape)

print(datasets.feature_names)

print(datasets.DESCR)  
x_train, x_test, y_train, y_test = train_test_split(x,y,
                train_size=0.7,
                shuffle=True,
                random_state=44)

#2. 모델구성
model = Sequential()
model.add(Dense(13, input_dim=10))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일 훈련 
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=150 , batch_size=33)

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_predict)
print('r2_score : ', r2)


# -------------------------------------------------------------
# 2022-06-27 






