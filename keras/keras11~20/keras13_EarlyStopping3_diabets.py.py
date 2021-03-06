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
        train_size=0.84, shuffle=True, random_state=100)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=10))
model.add(Dense(30))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(50))
model.add(Dense(1))

#3. 컴파일 , 훈련
model.compile(loss='mse', optimizer='adam')     # 회귀 모델의 대표적인 평가 지표 중에 하나 == R2(R제곱) R2수치가 높을수로 좋다 
model.fit(x_train, y_train, epochs=210, batch_size=10,
          validation_split=0.45)

from tensorflow.python.keras.callbacks import EarlyStopping
earlystopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, # mode='min'뿐아니라 max도 있음  디폴드값 찾아볼것 모르면 오토 
              restore_best_weights=True)  # < - 검색해서 정리할것 (파라미터를 적용을 시켯다 내가 하고싶은데로)
             # 모니터로 보겟다 vla_loss / patience 참다 10번 / mode = 'min'  최솟값을 verbose=1
             # 깃허브 참조 
             # 이름을 짓는다 earlystopping 변수는 첫번째를 소문자로 


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)  
from sklearn.metrics import r2_score         # metrics 행렬 
r2 = r2_score(y_test, y_predict)
print('r2score : ', r2)

