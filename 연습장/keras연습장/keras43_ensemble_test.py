
# rkas43 1 

# rkas43 1 

#1. data 

import numpy as np
x1_datasets = np.array([range(100), range(301, 401)]) # 삼성전자 증가 , 하이닉스 증가,  첫번째 데이터 셋 
x2_datasets = np.array([range(101, 201), range(411, 511), range(150,250)]) #  원유, 돈육, 밀
x3_datasets = np.array([range(100,200), range(1301, 1401)]) # 우리반 아이큐 , 우리반 키 
x1 = np.transpose(x1_datasets)
x2 = np.transpose(x2_datasets)
x3 = np.transpose(x3_datasets)

print(x1.shape, x2.shape, x3.shape)  #  ( 100,2 ) ( 100,3 ) (100, 2)

y1 = np.array(range(2001, 2101)) # 금리  ( 100 , )
y2 = np.array(range(201, 301)) # 환율 ( 100 , )

print(y1.shape)
print(y2.shape)

from sklearn.model_selection import train_test_split

# x1_train, x1_test, x2_train, x_2test, y_train, y_test = train_test_split(
#     x1, x2, y, train_size=0.8, random_state=55)


x1_train, x1_test, y1_train, y1_test = train_test_split(x1,y1,
    train_size=0.8, shuffle=True, random_state=55 )
x2_train, x2_test, y1_train, y1_test = train_test_split(x2,y1,
    train_size=0.8, shuffle=True, random_state=55)
x3_train, x3_test, y1_train, y1_test = train_test_split(x3,y1,
    train_size=0.8, shuffle=True, random_state=55)

x1_train, x1_test, y2_train, y2_test = train_test_split(x1,y2,
    train_size=0.8, shuffle=True, random_state=55 )
x2_train, x2_test, y2_train, y2_test = train_test_split(x2,y2,
    train_size=0.8, shuffle=True, random_state=55)
x3_train, x3_test, y2_train, y2_test = train_test_split(x3,y2,
    train_size=0.8, shuffle=True, random_state=55)


print(x1_train.shape, x1_test.shape)  # (80, 2) (20, 2)
print(x2_train.shape, x2_test.shape)  # (80, 3) (20, 3)
print(x3_train.shape, x3_test.shape)  # (80, 2) (20, 2)
print(y1_train.shape, y1_test.shape)  # (80,) (20,)
print(y2_train.shape, y2_test.shape)  # (80,) (20,)


#2. model
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Input

#2 - 모델 1번째

input1 = Input(shape=(2,))  #  인풋쉐이프
dense1 = Dense(100, activation='relu', name='dj1')(input1)
dense2 = Dense(100, activation='relu', name='dj2')(dense1)
dense3 = Dense(100, activation='relu', name='dj3')(dense2)
output1 = Dense(10, activation='relu', name='out_dj1')(dense3)

# 2-2 모델 2번째
input2 = Input(shape=(3,))  #  인풋쉐이프
dense11 = Dense(100, activation='relu', name='dj11')(input2)
dense12 = Dense(100, activation='relu', name='dj12')(dense11)
dense13 = Dense(100, activation='relu', name='dj13')(dense12)
dense14 = Dense(100, activation='relu', name='dj14')(dense13)
output2 = Dense(10, activation='relu', name='out_dj2')(dense14)

# 2 - 3 모델 3번째 
input3 = Input(shape=(2,))
dense111 = Dense(100, activation='relu', name='dj111')(input3)
dense112 = Dense(100, activation='relu', name='dj112')(dense111)
dense113 = Dense(100, activation='relu', name='dj113')(dense112)
dense114 = Dense(100, activation='relu', name='dj114')(dense113)
output3 = Dense(10, activation='relu', name='out_dj3')(dense114)

# concatenate 
from tensorflow.python.keras.layers import concatenate, Concatenate   # 사슬 같이 잇다
merge1 = concatenate([output1, output2, output3], name='mg1') # 두개의 아웃풋이 합쳐진 하나의 레이어 층 
merge2 = Dense(80, activation='relu', name='mg2')(merge1)
merge3 = Dense(50, name='mg3')(merge2)
merge4 = Dense(30, name='mg4')(merge3)
last_output1 = Dense(1, name='last1')(merge4)

# # 2-4, output model 1  
# output41 = Dense(10)(last_output1)
# output42 = Dense(10)(output41)
# last_output2 = Dense(1)(output42)

# # 2-4 output model 2
# output51 = Dense(110)(last_output1)
# output52 = Dense(110)(output51)
# output53 = Dense(110)(output52)
# last_output3 = Dense(1)(output53)

# model = Model(input=[input1, input2, input3], output=[last_output2, last_output3])
# model.summary()

merge12 = Dense(80, activation='relu', name='mg12')(merge1)
merge13 = Dense(50, name='mg13')(merge12)
merge14 = Dense(30, name='mg14')(merge13)
last_output2 = Dense(1, name='last2')(merge4)
model = Model(inputs=[input1, input2, input3],outputs=[last_output1, last_output2]) # 모델의 정의가 됨

model.summary()


# 맹그러봐

import time
from tensorflow.python.keras.callbacks import EarlyStopping

start_time = time.time()

earlystopping = EarlyStopping(monitor='val_loss', patience=200, mode='min', verbose=1,
                              restore_best_weights=True)

model.compile(loss = 'mse', optimizer = 'adam',
              metrics = ['accuracy'])

a = model.fit([x1_train, x2_train, x3_train], [y1_train, y2_train], epochs=2250, batch_size=200,
              validation_split=0.2,
              callbacks= [earlystopping], verbose=1)

end_time = time.time()-start_time
print(a)
print(a.history['val_loss'])

# 4. evaluate , perdict

loss1 = model.evaluate([x1_test, x2_test, x3_test], y1_test)
loss2 = model.evaluate([x1_test, x2_test, x3_test], y2_test)
print('loss : ', loss1)
print('loss : ', loss2)
y1_predict, y2_predict = model.predict([x1_test, x2_test, x3_test])

from sklearn.metrics import r2_score

r2_1 = r2_score(y1_test, y1_predict)
r2_2 = r2_score(y2_test, y2_predict)

print('r2score : ', r2_1)
print('r2score : ', r2_2)
print('loss : ', loss1)
print('loss : ', loss2)
print('걸린시간 : ', end_time)
print('keras43_ensemble2.py')



# loss :  [3240110.75, 0.0246577151119709, 3240110.75, 0.0, 0.0]
# loss :  [3240164.25, 3240164.25, 0.0029009433928877115, 0.0, 0.0]
# r2score :  0.9999789438800714
# r2score :  0.9999975227790266
# loss :  [3240110.75, 0.0246577151119709, 3240110.75, 0.0, 0.0]
# loss :  [3240164.25, 3240164.25, 0.0029009433928877115, 0.0, 0.0]
# 걸린시간 :  124.46833348274231
# keras43_ensemble2.py

















#1. data 
'''
import numpy as np
x1_datasets = np.array([range(100), range(301, 401)]) # 삼성전자 증가 , 하이닉스 증가,  첫번째 데이터 셋 
x2_datasets = np.array([range(101, 201), range(411, 511), range(150,250)]) #  원유, 돈육, 밀
x3_datasets = np.array([range(100,200), range(1301, 1401)]) # 우리반 아이큐 , 우리반 키 
x1 = np.transpose(x1_datasets)
x2 = np.transpose(x2_datasets)
x3 = np.transpose(x3_datasets)

print(x1.shape, x2.shape, x3.shape)  #  ( 100,2 ) ( 100,3 ) (100, 2)

y1 = np.array(range(2001, 2101)) # 금리  ( 100 , )
y2 = np.array(range(201, 301)) # 환율 ( 100 , )

print(y1.shape)
print(y2.shape)

from sklearn.model_selection import train_test_split

# x1_train, x1_test, x2_train, x_2test, y_train, y_test = train_test_split(
#     x1, x2, y, train_size=0.8, random_state=55)


x1_train, x1_test, y1_train, y1_test = train_test_split(x1,y1,
    train_size=0.9, shuffle=True, random_state=55 )
x2_train, x2_test, y1_train, y1_test = train_test_split(x2,y1,
    train_size=0.9, shuffle=True, random_state=55)
x3_train, x3_test, y1_train, y1_test = train_test_split(x3,y1,
    train_size=0.9, shuffle=True, random_state=55)

x1_train, x1_test, y2_train, y2_test = train_test_split(x1,y2,
    train_size=0.9, shuffle=True, random_state=55 )
x2_train, x2_test, y2_train, y2_test = train_test_split(x2,y2,
    train_size=0.9, shuffle=True, random_state=55)
x3_train, x3_test, y2_train, y2_test = train_test_split(x3,y2,
    train_size=0.9, shuffle=True, random_state=55)


print(x1_train.shape, x1_test.shape)  # (80, 2) (20, 2)
print(x2_train.shape, x2_test.shape)  # (80, 3) (20, 3)
print(x3_train.shape, x3_test.shape)  # (80, 2) (20, 2)
print(y1_train.shape, y1_test.shape)  # (80,) (20,)
print(y2_train.shape, y2_test.shape)  # (80,) (20,)


from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from sklearn.preprocessing import MaxAbsScaler, RobustScaler
# scaler = RobustScaler()
# scaler = MaxAbsScaler()

# scaler = MinMaxScaler()
scaler = StandardScaler()

scaler.fit(x1_train)
x_train = scaler.transform(x1_train)
x_test = scaler.transform(x1_test)
print(np.min(x_train))   # 0.0
print(np.max(x_train))   # 0.0 컬럼별로 나누어주어야 한다
print(np.min(x_test))
print(np.max(x_test))

scaler.fit(x2_train)
x_train = scaler.transform(x2_train)
x_test = scaler.transform(x2_test)
print(np.min(x_train))   # 0.0
print(np.max(x_train))   # 0.0 컬럼별로 나누어주어야 한다
print(np.min(x_test))
print(np.max(x_test))

scaler.fit(x3_train)
x_train = scaler.transform(x3_train)
x_test = scaler.transform(x3_test)
print(np.min(x_train))   # 0.0
print(np.max(x_train))   # 0.0 컬럼별로 나누어주어야 한다
print(np.min(x_test))
print(np.max(x_test))


#2. model
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Input, Dropout

#2 - 모델 1번째

input1 = Input(shape=(2,))  #  인풋쉐이프
dense1 = Dense(180, activation='relu', name='dj1')(input1)
dense2 = Dense(180, activation='relu', name='dj2')(dense1)
dense3 = Dense(180, activation='relu', name='dj4')(dense2)
output1 = Dense(30, activation='relu', name='out_dj1')(dense3)

# 2-2 모델 2번째
input2 = Input(shape=(3,))  #  인풋쉐이프
dense11 = Dense(180, activation='relu', name='dj11')(input2)
dense12 = Dense(180, activation='relu', name='dj12')(dense11)
dense13 = Dense(180, activation='relu', name='dj13')(dense12)
dense14 = Dense(180, activation='relu', name='dj15')(dense13)
output2 = Dense(30, activation='relu', name='out_dj2')(dense14)

# 2 - 3 모델 3번째 
input3 = Input(shape=(2,))
dense111 = Dense(180, activation='relu', name='dj111')(input3)
dense112 = Dense(180, activation='relu', name='dj112')(dense111)
dense113 = Dense(180, activation='relu', name='dj113')(dense112)
dense114 = Dense(180, activation='relu', name='dj114')(dense113)
output3 = Dense(30, activation='relu', name='out_dj3')(dense114)


from tensorflow.python.keras.layers import concatenate, Concatenate   # 사슬 같이 잇다

merge1 = concatenate([output1, output2, output3], name='mg1') # 두개이상의 아웃풋이 합쳐진 하나의 레이어 층 
merge2 = Dense(200, activation='relu', name='mg2')(merge1)
merge3 = Dense(100, name='mg3')(merge2)
merge4 = Dense(50, name='mg4')(merge3)
last_output1 = Dense(1, name='last1')(merge4)

merge12 = Dense(200, activation='relu', name='mg12')(merge1)
merge13 = Dense(100, name='mg13')(merge12)
merge14 = Dense(50, name='mg14')(merge13)
last_output2 = Dense(1, name='last2')(merge4)

model = Model(inputs=[input1, input2, input3],outputs=[last_output1, last_output2]) # 모델의 정의가 됨


model.summary()


# 맹그러봐

import time
from tensorflow.python.keras.callbacks import EarlyStopping

start_time = time.time()

earlystopping = EarlyStopping(monitor='val_loss', patience=200, mode='min', verbose=1,
                              restore_best_weights=True)

model.compile(loss = 'mse', optimizer = 'adam',
              metrics = ['accuracy'])

a = model.fit([x1_train, x2_train, x3_train], [y1_train, y2_train], epochs=2250, batch_size=200,
              validation_split=0.3,
              callbacks= [earlystopping], verbose=1)

end_time = time.time()-start_time
print(a)
print(a.history['val_loss'])

# 4. evaluate , perdict

loss1 = model.evaluate([x1_test, x2_test, x3_test], y1_test)
loss2 = model.evaluate([x1_test, x2_test, x3_test], y2_test)
print('loss : ', loss1)
print('loss : ', loss2)
y1_predict, y2_predict = model.predict([x1_test, x2_test, x3_test])

from sklearn.metrics import r2_score

r2_1 = r2_score(y1_test, y1_predict)
r2_2 = r2_score(y2_test, y2_predict)

print('r2score : ', r2_1)
print('r2score : ', r2_2)
print('loss : ', loss1)
print('loss : ', loss2)
print('걸린시간 : ', end_time)
print('keras43_ensemble2.py')



# loss :  [3240110.75, 0.0246577151119709, 3240110.75, 0.0, 0.0]
# loss :  [3240164.25, 3240164.25, 0.0029009433928877115, 0.0, 0.0]
# r2score :  0.9999789438800714
# r2score :  0.9999975227790266
# loss :  [3240110.75, 0.0246577151119709, 3240110.75, 0.0, 0.0]
# loss :  [3240164.25, 3240164.25, 0.0029009433928877115, 0.0, 0.0]
# 걸린시간 :  124.46833348274231
# keras43_ensemble2.py


# scaler 스텐다드 

# loss :  [3239641.5, 0.11894842237234116, 3239641.5, 0.0, 0.0]
# loss :  [3240381.5, 3240381.5, 0.017849216237664223, 0.0, 0.0]
# r2score :  0.9998984256148252
# r2score :  0.9999847579075597
# loss :  [3239641.5, 0.11894842237234116, 3239641.5, 0.0, 0.0]
# loss :  [3240381.5, 3240381.5, 0.017849216237664223, 0.0, 0.0]
# 걸린시간 :  80.9739158153534
# keras43_ensemble2.py



# loss :  [3240004.5, 0.04984153062105179, 3240004.5, 0.0, 0.0]
# loss :  [3240316.75, 3240316.75, 0.001473411452025175, 0.0, 0.0]
# r2score :  0.9999428553896688
# r2score :  0.9999983106953964
# loss :  [3240004.5, 0.04984153062105179, 3240004.5, 0.0, 0.0]
# loss :  [3240316.75, 3240316.75, 0.001473411452025175, 0.0, 0.0]
# 걸린시간 :  136.0015754699707
# keras43_ensemble2.py


# loss :  [3240125.25, 0.01998278871178627, 3240125.25, 0.0, 0.0]
# loss :  [3240139.25, 3240139.25, 0.01299243699759245, 0.0, 0.0]
# r2score :  0.9999770892118128
# r2score :  0.9999851038324389
# loss :  [3240125.25, 0.01998278871178627, 3240125.25, 0.0, 0.0]
# loss :  [3240139.25, 3240139.25, 0.01299243699759245, 0.0, 0.0]
# 걸린시간 :  136.7971305847168
# keras43_ensemble2.py



'''



