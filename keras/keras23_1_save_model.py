from matplotlib import font_manager
from sklearn. datasets import load_boston        # 사이킷 런에는 예제 문제가 있고 데이터가 있다
import numpy as np 
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

# 1. 데이터

datasets = load_boston()              # load_boston 에서 x, y  데이터를 추출 한다 
x = datasets.data          #  ( x 의 데이터 로 )
y = datasets.target        #  ( y 의 값을 구한다 )

print(x)     # ( 보스턴 집 값을 위한 데이터 ) 

             # 연산자 찾아보기 ( 4.7410e-02)  e-ㅁㅁ 는 소숫점 뒤 0의 갯수   
             #                ( 3.9690e+02 ) e+ㅁㅁ 는 소숫점 앞 0의 갯수 
             # 데이터 전처리 0 ~ 1 까지 0 50 100 을 0 0.5 1 로 바꿀때 y값은 동일하게 가리킨다 
print(y)     # ( 보스턴 집 값을 위한 데이터를 사용하여 나온 보스턴 집 값 )

print(x.shape, y.shape)     # x = (506, 13)  y [506개의 스칼라에 1 벡터] = (506,)  506개의 데이터 13개의 컬럼 인풋 13 아웃풋 1

print(datasets.feature_names)       # ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'        
                                    # 'B' 'LSTAT']  범죄 세금 'b' 흑인  [ 조심 ] 오류 찾아볼것 

print(datasets.DESCR)         #피쳐 아주중요 따로 찾아볼것 
x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.7, shuffle=True, random_state=55)


# scaler = RobustScaler()
# scaler = MaxAbsScaler()

scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# print(np.min(x_train))   # 0.0
# print(np.max(x_train))   # 0.0 컬럼별로 나누어주어야 한다
# print(np.min(x_test))
# print(np.max(x_test))


# 2. 모델구성


model = Sequential()
model.add(Dense(64, input_dim=13))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))
model.summary()

model.save('./_save/keras23_1_save_model.h5')  # . 작업경로 / _save 밑에 / 만들겟다 이 파일을 ( model. save 모델만 세이브 )



'''

#3. 컴파일 , 훈련

start_time = time.time()

model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping
earlystopping = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, 
              restore_best_weights=True)

a = model.fit(x_train, y_train, epochs=100, batch_size=25,
                validation_split=0.25,
                callbacks=[earlystopping],
)


print(a)
print(a.history['loss']) # 대괄호로 loss , val loss 값 출력 가능

end_time = time.time() - start_time

# 4 . 평가 예측 
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

'''

