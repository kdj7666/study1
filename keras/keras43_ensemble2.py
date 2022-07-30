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

y = np.array(range(2001, 2101)) # 금리  ( 100 , )
print(y.shape)
from sklearn.model_selection import train_test_split

# x1_train, x1_test, x2_train, x_2test, y_train, y_test = train_test_split(
#     x1, x2, y, train_size=0.8, random_state=55)


x1_train, x1_test, y_train, y_test = train_test_split(x1,y,
    train_size=0.8, shuffle=True, random_state=55 )
x2_train, x2_test, y_train, y_test = train_test_split(x2,y,
    train_size=0.8, shuffle=True, random_state=55)
x3_train, x3_test, y_train, y_test = train_test_split(x3,y,
    train_size=0.8, shuffle=True, random_state=55)



print(x1_train.shape, x1_test.shape)  # (80, 2) (20, 2)
print(x2_train.shape, x2_test.shape)  # (80, 3) (20, 3)
print(x3_train.shape, x3_test.shape)  # (80, 2) (20, 2)
print(y_train.shape, y_test.shape)  # (70,) (30,)

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


from tensorflow.python.keras.layers import concatenate, Concatenate   # 사슬 같이 잇다
merge1 = concatenate([output1, output2, output3], name='mg1') # 두개의 아웃풋이 합쳐진 하나의 레이어 층 
merge2 = Dense(80, activation='relu', name='mg2')(merge1)
merge3 = Dense(50, name='mg3')(merge2)
merge4 = Dense(30, name='mg4')(merge3)
last_output = Dense(1, name='last')(merge4)

model = Model(inputs=[input1, input2, input3],outputs=last_output) # 모델의 정의가 됨

model.summary()

# 맹그러봐 

import time
from tensorflow.python.keras.callbacks import EarlyStopping

start_time = time.time()

earlystopping = EarlyStopping(monitor='val_loss', patience=200, mode='min', verbose=1,
                              restore_best_weights=True)

model.compile(loss = 'mse', optimizer = 'adam',
              metrics = ['accuracy'])

a = model.fit([x1_train, x2_train, x3_train], y_train, epochs=1250, batch_size=200,
              validation_split=0.2,
              callbacks= [earlystopping], verbose=1)

end_time = time.time()-start_time
print(a)
print(a.history['val_loss'])

# 4. evaluate , perdict

loss = model.evaluate([x1_test,x2_test, x3_test], y_test)
print('loss : ', loss)

y_predict = model.predict([x1_test,x2_test, x3_test])

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_predict)

print('r2score : ', r2)
print('loss : ', loss)
print('걸린시간 : ', end_time)
print('keras43_ensemble2.py')

# acc = accuracy_score(y_test, y_predict)

# print('acc.score : ', acc)


# loss :  [0.21121379733085632, 0.0]
# 걸린시간 :  48.75331115722656
# loss :  [0.21121379733085632, 0.0]


# loss :  [0.6263588666915894, 0.0]
# r2score :  0.9994651295135405
# 걸린시간 :  53.56546974182129
# loss :  [0.6263588666915894, 0.0]



# loss :  [0.04475238174200058, 0.0]
# r2score :  0.9999617843168173
# 걸린시간 :  63.443485498428345
# loss :  [0.04475238174200058, 0.0]

