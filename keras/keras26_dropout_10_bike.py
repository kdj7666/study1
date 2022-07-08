import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import nan_euclidean_distances, r2_score, mean_squared_error
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

#1. data  # 10번 경로  +는 문자가 연결이 된다
path = './_data/bike/' 
train_set = pd.read_csv(path + 'train.csv', 
                        index_col=0)         #index_col 1번째는 id 행의 이름이기때문에 계산 ㄴ
print(train_set)             
print(train_set.shape)       # 1459개의 열과 10개의 컬럼  (1459,10)

test_set = pd.read_csv(path + 'test.csv',
                       index_col=0)     #  예측에서 프레딕트로 쓸것이다

submission = pd.read_csv(path + 'samplesubmission.csv')

print(test_set)
print(test_set.shape)   # 715개의 열과 9개의 컬럼  (715,9)

print(train_set.columns) 
print(train_set.info())       # 컬럼에 대한 내용이 디테일하게 나온다       ( Non-Null Count ) 이빨이 빠졋다 데이터가 빠졋다  [ 결측치 ] 데이터 전처리에 아주 중요 / [이상치]라는 데이터도 있다 나중에 
print(train_set.describe())               #  describe 묘사하다 서술하다  # 최솟값 최댓값 등 확인       pd 좀더 찾아보기 중요

#### 결측치 처리 1. 제거####
print(train_set.isnull().sum())
train_set = train_set.dropna()
print(train_set.isnull().sum())
print(train_set.shape)
#############################
x = train_set.drop(['count','casual','registered',], axis=1)   # drop 날리다 카운트라는 줄을 날릴것이다 소숫점이 1개 
print(x)
print(x.columns)
print(x.shape) #  ( 1459 , 9 )

y = train_set['count']  # 이렇게 하면 빠진다 지금은 이정도 ( [ ] 대괄호를 잘못치면 다 틀린다 ) 나중에 반복
print(y)
print(y.shape)   # ( 1459 , ) # 벡터가 1개 그래서 최종 아웃풋 갯수는 1개   ( 여기까지가 데이터 )

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.989,
        shuffle=True,
        random_state=40)



# scaler = MinM# from sklearn.preprocessing import MinMaxScaler, StandardScaler
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

model = Sequential()
model.add(Dense(90, input_dim=8))          # 행 무시 열 우선 필수 
model.add(Dense(80, activation='relu'))
model.add(Dense(80, activation='relu'))  
model.add(Dropout(0.3))
model.add(Dense(50))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.3))  
model.add(Dense(50))   
model.add(Dense(1))

# input1 = Input(shape=(8,))
# dense1 = Dense(10)(input1)
# dense2 = Dense(5, activation='relu')(dense1)
# dense3 = Dense(3, activation='relu')(dense2)
# output1 = Dense(1)(dense3)
# model = Model(inputs=input1, outputs=output1)

# model.save('./_save/k23_smm_bike.h5')




#3. 컴파일 훈련

start_time = time.time()
model.compile(loss='mae', optimizer = 'adam')   
# 평가지표는 프레딕트 결과값 어쩌구 저쩌구 해서 mse 로 가능 비슷하면 된다 

import datetime
date = datetime.datetime.now()
print(date)
date = date.strftime("%m%d_%H%M")
print(date)

filepath = './_ModelCheckPoint/k24/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

earlystopping = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, # mode='min'뿐아니라 max도 있음  디폴드값 찾아볼것 모르면 오토 
              restore_best_weights=True)  # < - 검색해서 정리할것 (파라미터를 적용을 시켯다 내가 하고싶은데로)
             # 모니터로 보겟다 vla_loss / patience 참다 10번 / mode = 'min'  최솟값을 verbose=1
             # 깃허브 참조 
             # 이름을 짓는다 earlystopping 변수는 첫번째를 소문자로 
             

mcp = ModelCheckpoint(monitor='val_loss', node='auto', verbose=1,
                      save_best_only=True,
                      filepath= "".join([filepath, 'k24_bike', date, '_', filename]))
             
             
a = model.fit(x_train, y_train, epochs=300, batch_size=100,
          validation_split=0.2,
          callbacks = [earlystopping, mcp],
          verbose=1)   # a 대신에 hist 라고 쓰임 콜백을 하겠다 얼리 스탑잉을               



print(a)
print(a.history['val_loss']) # 대괄호로 loss , val loss 값 출력 가능

end_time = time.time()-start_time

#4. 평가 예측

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
     return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

y_summit = model.predict(test_set)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2scoer :', r2)
print('걸린시간: ', end_time)
model.summary()
# print(y_summit)
# print(y_summit.shape) # (715,1)

######################## .to_csv()를 사용해서 아이디값 안됨 카운트값 순서대로 
### submission.csv를 완성하시오 !!! ( 과제 겸 실습 )
# dataframe = pd.DataFrame(y_summit)
# dataframe.to_csv('.csv')

submission['count'] = y_summit        
# submission = submission.fillna(submission.mean())
submission.to_csv(path + 'samplesubmission.csv', index=False)

# 없음 
# loss :  121.881103515625
# RMSE :  180.05878184979784
# r2scoer : 0.23099381454002577
# 걸린시간:  69.54618120193481

# min max 
# loss :  106.98854064941406
# RMSE :  152.24571877902716
# r2scoer : 0.45021688769230406
# 걸린시간:  68.98352360725403

# standard
# loss :  108.47122955322266
# RMSE :  156.69548901058062
# r2scoer : 0.4176096023426177
# 걸린시간:  67.29661893844604

# RobustScaler
# loss :  107.62580108642578
# RMSE :  161.50663579836677
# r2scoer : 0.38129737402663966
# 걸린시간:  94.43533134460449

# MaxAbsScaler
# loss :  108.90987396240234
# RMSE :  160.40696403064692
# r2scoer : 0.3896939769862434
# 걸린시간:  98.87416362762451

# maxabsscaler 2회 좋아진듯 
# loss :  106.01976013183594
# RMSE :  158.7860530033222
# r2scoer : 0.4019659326124505
# 걸린시간:  88.3551971912384

# 함수형 모델 
# loss :  123.52904510498047
# RMSE :  178.25396914979535
# r2scoer : 0.24633276410022464
# 걸린시간:  89.26311111450195

