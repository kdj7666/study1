import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

#1. data 
datasets = load_breast_cancer()
#print(datasets)
print(datasets.DESCR)                        
    # :Number of (행) Instances: 569      instances 행 
    # :Number of (열) Attributes: 30 numeric, predictive attributes and the class     569 , 30
#print(datasets.feature_names) # 컬럼의 이름 

x = datasets.data # x = datasets['data'] 이렇게도 사용 가능
y = datasets.target
print(x.shape, y.shape)  #   569 , 30     569 , 

print(x)
print(y)

print(datasets.DESCR)         #피쳐 아주중요 따로 찾아볼것 
x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.9, shuffle=True, random_state=100)

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

# 2. 모델구성

model = Sequential()
model.add(Dense(100, activation='linear', input_dim=30))
model.add(Dense(50, activation='sigmoid')) # activation 활성화 함수 이것으로 인해 결과값이 엄청좋아지고 안좋아진다 필수 ( 다시 공부할것 시그모이드 포함 )
model.add(Dense(50, activation='relu'))    #  레이어를 한정시키며 펑 터지는것을 방지한다 ( 찾아볼것 )
model.add(Dense(50, activation='relu')) # linear 선형  
model.add(Dense(1, activation='sigmoid')) # sigmoid = 0과 1이 아니고 0에서 1으로 한정 시킨다 ( 어떤값을 넣어도 0과 1사이로 표출이 된다 ) 반올림으로 0과 1로 
# 이진분류는 무조건 sigmoid 로 끝난다 그다음은 binary_crossentropy 로 쓴다 
#  activation relu 히든에서만 사용이 가능하다 relu 성능 아주 좋다 85% 성능향상 중요 


#3. 컴파일 , 훈련
start_time = time.time()
model.compile(loss='binary_crossentropy', optimizer='adam',   # 회기모델에서 accuracy , mae 둘다 가능 지표를 찾을 수 있음 
              metrics=['accuracy', 'mse'])    # 정확성  accuracy: 0.9474 / 94.74%     2개 이상은 리스트 형식 더 넣을수있음 프로그래스바에 표츌이 늘어남 
                             # 회귀 모델의 대표적인 평가 지표 중에 하나 == R2(R제곱) R2수치가 높을수로 좋다  ****** 중요 

                             # 2진분류로 sigmoid 를 쓸때에는 model.compile(loss='binary_crossentropy', optimizer='adam')      ****** 중요
                             # 로 한다   /   당분간 이거 하나쓴다 ( 나중에 바뀔 수 있음) 0과 1에 한해서                          ****** 중요 

earlystopping = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=1, # mode='min'뿐아니라 max도 있음  디폴드값 찾아볼것 모르면 오토 

              restore_best_weights=True)  # < - 검색해서 정리할것 (파라미터를 적용을 시켯다 내가 하고싶은데로)
             # 모니터로 보겟다 vla_loss / patience 참다 10번 / mode = 'min'  최솟값을 verbose=1
             # 깃허브 참조 
             # 이름을 짓는다 earlystopping 변수는 첫번째를 소문자로 
             
a = model.fit(x_train, y_train, epochs=1000, batch_size=50,
          validation_split=0.2,
          callbacks = [earlystopping],    # 이것도 리스트 형식이라는것 더 넣을수있음 
          verbose=1)   # a 대신에 hist 라고 쓰임 콜백을 하겠다 얼리 스탑잉을               


end_time = time.time()-start_time
#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
print(a.history['val_loss'])

y_predict = model.predict(x_test)  # 이 값이 54번 으로
y_predict = y_predict.flatten()
y_predict = np.where(y_predict > 0.5, 1 , 0)
print(y_predict)

#### [ 과제 1.] accuracy_score 완성 시키기


from sklearn.metrics import r2_score, accuracy_score         # metrics 행렬 
# r2 = r2_score(y_test, y_predict)
acc = accuracy_score(y_test, y_predict)
print('acc.score : ', acc)
# print(y_predict)
print('걸린시간 : ', end_time)

# 없음
# acc.score :  0.9473684210526315
# 걸린시간 :  3.4547674655914307

# min max 
# acc.score :  0.9649122807017544
# 걸린시간 :  7.675487518310547

# standard
# acc.score :  0.9649122807017544
# 걸린시간 :  3.9334876537323
