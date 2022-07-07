# https://dacon.io/competitions/open/235576/overview/description

# 데이콘 따릉이 문제풀이 
import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import nan_euclidean_distances, r2_score, mean_squared_error
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

#1. data  # 10번 경로  +는 문자가 연결이 된다
path = './_data/ddarung/' 
train_set = pd.read_csv(path + 'train.csv', 
                        index_col=0)         #index_col 1번째는 id 행의 이름이기때문에 계산 ㄴ
print(train_set)             
print(train_set.shape)       # 1459개의 열과 10개의 컬럼  (1459,10)

test_set = pd.read_csv(path + 'test.csv',
                       index_col=0)     #  예측에서 프레딕트로 쓸것이다

submission = pd.read_csv(path + 'submission.csv')

print(test_set)
print(test_set.shape)   # 715개의 열과 9개의 컬럼  (715,9)

print(train_set.columns) 
print(train_set.info())       # 컬럼에 대한 내용이 디테일하게 나온다       ( Non-Null Count ) 이빨이 빠졋다 데이터가 빠졋다  [ 결측치 ] 데이터 전처리에 아주 중요 / [이상치]라는 데이터도 있다 나중에 
print(train_set.describe())               #  describe 묘사하다 서술하다  # 최솟값 최댓값 등 확인       pd 좀더 찾아보기 중요

#### 결측치 처리 1. 제거####
print(train_set.isnull().sum())
train_set = train_set.dropna()
test_set = train_set.fillna(test_set.mean())
print(train_set.isnull().sum())
print(train_set.shape)
#############################
x = train_set.drop(['count'], axis=1)   # drop 날리다 카운트라는 줄을 날릴것이다 소숫점이 1개 
print(x)
print(x.columns)
print(x.shape) #  ( 1459 , 9 )

y = train_set['count']  # 이렇게 하면 빠진다 지금은 이정도 ( [ ] 대괄호를 잘못치면 다 틀린다 ) 나중에 반복
print(y)
print(y.shape)   # ( 1459 , ) # 벡터가 1개 그래서 최종 아웃풋 갯수는 1개   ( 여기까지가 데이터 )

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.7,
        shuffle=True,
        random_state=40)


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



# np.logical_or(x, y)
# print(x = train_set.info(x))
# print(train_set.dropna( subset['Age']))
# print(pd.isna('nan'))

#2. 모델구성

# model = Sequential()
# model.add(Dense(90, input_dim=9))          # 행 무시 열 우선 필수 
# model.add(Dense(80, activation='relu'))
# model.add(Dense(80, activation='relu'))  
# model.add(Dense(50, activation='relu'))
# model.add(Dense(50))  
# model.add(Dense(50))   
# model.add(Dense(1))


input1 = Input(shape=(9,))
dense1 = Dense(90)(input1)
dense2 = Dense(80, activation='relu')(dense1)
dense3 = Dense(80, activation='relu')(dense2)
dense4 = Dense(50, activation='relu')(dense3)
dense5 = Dense(50)(dense4)
dense6 = Dense(50)(dense5)
output1 = Dense(1, activation='softmax')(dense6)
model = Model(inputs=input1, outputs=output1)

# model.save('./_save/k23_smm_ddarung.h5')



#3. 컴파일 훈련

start_time = time.time()
model.compile(loss='categorical_crossentropy', optimizer = 'adam',metrics=['accuracy', 'mse'])        # 평가지표는 프레딕트 결과값 어쩌구 저쩌구 해서 mse 로 가능 비슷하면 된다 

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
earlystopping = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, # mode='min'뿐아니라 max도 있음  디폴드값 찾아볼것 모르면 오토 
              restore_best_weights=True)  # < - 검색해서 정리할것 (파라미터를 적용을 시켯다 내가 하고싶은데로)
             # 모니터로 보겟다 vla_loss / patience 참다 10번 / mode = 'min'  최솟값을 verbose=1
             # 깃허브 참조 
             # 이름을 짓는다 earlystopping 변수는 첫번째를 소문자로 


import datetime
date = datetime.datetime.now()
print(date)
date = date.strftime("%m%d_%H%M")
print(date)

filepath = './_ModelCheckPoint/k24/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'



mcp = ModelCheckpoint(monitor='val_loss', node='auto', verbose=1,
                      save_best_only=True,
                      filepath= "".join([filepath, 'k24_ddrung', date, '_', filename]))

a = model.fit(x_train, y_train, epochs=300, batch_size=300,
          validation_split=0.2,
          callbacks = [earlystopping, mcp],
          verbose=1)   # a 대신에 hist 라고 쓰임 콜백을 하겠다 얼리 스탑잉을               
end_time = time.time()-start_time
#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
print(a.history['val_loss'])

y_predict = model.predict(x_test)
y_predict = y_predict.flatten()
y_predict = np.where(y_predict > 0.5, 1 , 0)
print(y_predict)


from sklearn.metrics import r2_score, accuracy_score         # metrics 행렬 
r2 = r2_score(y_test, y_predict)
acc = accuracy_score(y_test, y_predict)
print('acc.score : ', acc)
# print(y_predict)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
     return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)
print('r2 score : ', r2)
print('걸린시간 : ', end_time)
model.summary()


# 없음 
# acc.score :  0.005012531328320802
# RMSE :  53.29223877166541
# r2 score :  -2.0120692960543685
# 걸린시간 :  5.358732223510742

# min max 
# acc.score :  0.005012531328320802
# RMSE :  47.09539614014186
# r2 score :  -2.0120825464542023
# 걸린시간 :  8.793152093887329

#  stendard
# acc.score :  0.005012531328320802
# RMSE :  47.473921540802074
# r2 score :  -2.0123980716002374
# 걸린시간 :  7.627747297286987

# 함수형 모델 
# acc.score :  0.005012531328320802
# RMSE :  60.02942897032984
# r2 score :  -2.0120755071792904
# 걸린시간 :  8.985129356384277

