import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

#1.데이터
datasets = load_iris()
print(datasets.DESCR)
#4가지를 가르쳐줄테니 50개의 3종류 꽃으 맞춰라
# :Number of Instances: 150 (50 in each of three classes)
# # y=         - class:
#                 - Iris-Setosa
#                 - Iris-Versicolour
#                 - Iris-Virginica
print(datasets.feature_names)
x = datasets['data']
y = datasets['target']
print(x)
print(y)
print(y.shape)

print(datasets.DESCR)         #피쳐 아주중요 따로 찾아볼것 
x_train,y_train,x_test,y_test = train_test_split(x,y,
        train_size=0.7, shuffle=True, random_state=55)

# 2. 모델구성

model = Sequential()
model.add(Dense(26, input_dim=4))
model.add(Dense(40))
model.add(Dense(50))
model.add(Dense(60))
model.add(Dense(1))

#3. 컴파일 , 훈련
model.compile(loss='mse', optimizer='adam')     # 회귀 모델의 대표적인 평가 지표 중에 하나 == R2(R제곱) R2수치가 높을수로 좋다 

from tensorflow.python.keras.callbacks import EarlyStopping
earlystopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, # mode='min'뿐아니라 max도 있음  디폴드값 찾아볼것 모르면 오토 
              restore_best_weights=True)  # < - 검색해서 정리할것 (파라미터를 적용을 시켯다 내가 하고싶은데로)
             # 모니터로 보겟다 val_loss / patience 참다 10번 / mode = 'min'  최솟값을 verbose=1
             # 깃허브 참조 
             # 이름을 짓는다 earlystopping 변수는 첫번째를 소문자로 
             
a = model.fit(x_train, y_train, epochs=1000, batch_size=1,
          validation_split=0.2,
          callbacks = [earlystopping],
          verbose=1)   # a 대신에 hist 라고 쓰임 콜백을 하겠다 얼리 스탑잉을               

# end_time = time.time
print(a)
print(a.history['val_loss']) # 대괄호로 loss , val loss 값 출력 가능

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)  # 이 값이 54번 으로 
from sklearn.metrics import r2_score         # metrics 행렬 
r2 = r2_score(y_test, y_predict)
print('r2score : ', r2)

#그림그릴거야 # 데이터 값의 시각화를 해달라 중복으로 납둘것 

import matplotlib.pyplot as plt

from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name() # 49 50 51 세줄은 한글때문에 필요한것 * 한국인 필수
rc('font', family=font_name)

plt.figure(figsize=(15,10)) # plt.show 의 칸 가로의 길이 세로의 길이 
plt.plot(a.history['lva_loss'], marker='.', c='red', label='loss') # 곡선의 꺾임 marker . c = 색깔 red label loss 

#그림그릴거야 # 데이터 값의 시각화를 해달라 시각화 잘할것 보고서 작성에도 중요  

plt.plot(a.history['val_loss'], marker=',', c='blue', label= 'val_loss')
plt.grid()  # plt.show 의 그래프에 눈금을 그린다 
plt.title('val_loss') # 그래프 위의 제목 타이틀 
plt.ylabel('loss')    # 55번과 57번 색과 그래프의 선이 다름 표현해달라 
plt.xlabel('epochs')  # x는 epochs 수치를 표현해달라 
plt.legend(loc='upper right') # 라벨값 위치 생략시 빈자리에 생성 # loc='upper right' 상단 오른쪽 
plt.show() # 이 그래프를 보여달라 


