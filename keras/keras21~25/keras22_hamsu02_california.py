from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split # 앞에 소문자는 변수 또는 함수 함수로 인식
import numpy as np
from sqlalchemy import Time
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler # 카멜케이스 문제는없다
from sklearn.preprocessing import MaxAbsScaler, RobustScaler # ( 찾아 본 후 정리 하고 keras20 파일 전부 적용 ) 둘중 이상치가 잘 골라지는게 있음 

# data
datasets = load_boston()
x = datasets.data
y = datasets['target']

print(x.shape)
print(y.shape)
# print(np.min(x)) # x의 최솟값 0.0
# print(np.max(x)) # x의 최댓값 711.0     711을 1로 잡고 최댓값이 형성
# x = (x - np.min(x)) / (np.max(x) - np.min(x)) # minmaxs 데이터를 통과시키면 0에서 1사이로 된다 ( 나중에 1~ 넣어볼것 ) 다시 찾아서 공부할것  
# print(x[:10])
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


# a = 0.1
# b = 0.2
# print(a + b)

# 2. 모델구성
# 3. 컴파일 훈련
# 4. 평가 예측

# 보스턴에 대해서 3가지 비교 

# 스케일러 하기전
# 민맥스
# 스탠다드 

# 3개 비교 

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

# model = Sequential()
# model.add(Dense(26, input_dim=13))
# model.add(Dense(40, activation='relu'))
# model.add(Dense(50, activation='relu'))
# model.add(Dense(60, activation='relu'))
# model.add(Dense(1, activation='relu'))

input1 = Input(shape=(13,))
dense1 = Dense(10)(input1)
dense2 = Dense(5, activation='relu')(dense1)
dense3 = Dense(3, activation='relu')(dense2)
output1 = Dense(1)(dense3)
model = Model(inputs=input1, outputs=output1)

model.save('./_save/k23_smm_california.h5')

# model = load_model('./_save/keras23_8.1_save_model.h5')


#3. 컴파일 , 훈련

start_time = time.time()

model.compile(loss='mse', optimizer='adam')     # 회귀 모델의 대표적인 평가 지표 중에 하나 == R2(R제곱) R2수치가 높을수로 좋다 

from tensorflow.python.keras.callbacks import EarlyStopping
earlystopping = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, # mode='min'뿐아니라 max도 있음  디폴드값 찾아볼것 모르면 오토 
              restore_best_weights=True)  # < - 검색해서 정리할것 (파라미터를 적용을 시켯다 내가 하고싶은데로)
             # 모니터로 보겟다 vla_loss / patience 참다 10번 / mode = 'min'  최솟값을 verbose=1
             # 깃허브 참조 
             # 이름을 짓는다 earlystopping 변수는 첫번째를 소문자로 
             
a = model.fit(x_train, y_train, epochs=300, batch_size=100,
          validation_split=0.2,
          callbacks = [earlystopping],
          verbose=1)   # a 대신에 hist 라고 쓰임 콜백을 하겠다 얼리 스탑잉을               


print(a)
print(a.history['val_loss']) # 대괄호로 loss , val loss 값 출력 가능

end_time = time.time()-start_time


model.save('./_save/keras23_8_save_model.h5')

# model - load_model('./_save/keras23_8_save_model.h5')

#4. 평가, 예측

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)


y_predict = model.predict(x_test)  # 이 값이 54번 으로 
from sklearn.metrics import r2_score         # metrics 행렬 
r2 = r2_score(y_test, y_predict)
print('r2score : ', r2)
print('걸린시간 :', end_time)

# 없음
# loss :  606.0347900390625
# r2score :  -7.37129308573221
# 걸린시간 : 4.769369125366211

# min max 
# loss :  606.0347900390625
# r2score :  -7.37129308573221
# 걸린시간 : 4.839867830276489

# Standard 
# loss :  10.352697372436523
# r2score :  0.8569958973624809
# 걸린시간 : 13.034252882003784

# RobustScaler
# loss :  8.837111473083496
# r2score :  0.8779310233135745
# 걸린시간 : 10.493794202804565

# MaxAbsScaler
# loss :  606.0347900390625
# r2score :  -7.37129308573221
# 걸린시간 : 3.4653944969177246

# 함수형 모델 
# loss :  29.40367317199707
# r2score :  0.5938405165029819
# 걸린시간 : 10.682676076889038




