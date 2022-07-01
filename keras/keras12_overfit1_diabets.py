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
model.compile(loss='mae', optimizer='adam')     # 회귀 모델의 대표적인 평가 지표 중에 하나 == R2(R제곱) R2수치가 높을수로 좋다 
a = model.fit(x_train, y_train, epochs=210, batch_size=10,
          validation_split=0.45)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)  

from sklearn.metrics import r2_score         # metrics 행렬 
r2 = r2_score(y_test, y_predict)
print('r2score : ', r2)

#그림그릴거야 # 데이터 값의 시각화를 해달라 중복으로 납둘것 

import matplotlib.pyplot as plt

from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name() # 49 50 51 세줄은 한글때문에 필요한것 * 한국인 필수
rc('font', family=font_name)

plt.figure(figsize=(5,5)) # plt.show 의 칸 가로의 길이 세로의 길이 
plt.plot(a.history['loss'], marker='.', c='red', label='loss') # 곡선의 꺾임 marker . c = 색깔 red label loss 

#그림그릴거야 # 데이터 값의 시각화를 해달라 시각화 잘할것 보고서 작성에도 중요  

plt.plot(a.history['val_loss'], marker=',', c='blue', label= 'val_loss')
plt.grid()  # plt.show 의 그래프에 눈금을 그린다 
plt.title('지각빈도') # 그래프 위의 제목 타이틀 
plt.ylabel('loss')    # 55번과 57번 색과 그래프의 선이 다름 표현해달라 
plt.xlabel('epochs')  # x는 epochs 수치를 표현해달라 
plt.legend(loc='upper right') # 라벨값 위치 생략시 빈자리에 생성 # loc='upper right' 상단 오른쪽 
plt.show() # 이 그래프를 보여달라 

# <keras.callbacks.History object at 0x0000015F32465700> 케라스 파일 콜백에 있는 히스토리의 / 메모리 저장된 메모리 주소 

# loss :  39.43647003173828 
# r2score :  0.5532839102959444