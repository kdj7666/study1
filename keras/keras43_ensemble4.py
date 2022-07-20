# rkas43 1 

#1. data 

import numpy as np
x1_datasets = np.array([range(100), range(301, 401)]) # 삼성전자 증가 , 하이닉스 증가,  첫번째 데이터 셋 

x1 = np.transpose(x1_datasets)


print(x1.shape)  #  ( 100,2 ) ( 100,3 ) (100, 2)

y1 = np.array(range(2001, 2101)) # 금리  ( 100 , )
y2 = np.array(range(201, 301)) # 환율 ( 100 , )

print(y1.shape)
print(y2.shape)

from sklearn.model_selection import train_test_split

# x1_train, x1_test, x2_train, x_2test, y_train, y_test = train_test_split(
#     x1, x2, y, train_size=0.8, random_state=55)


x1_train, x1_test, y1_train, y1_test = train_test_split(x1,y1,
    train_size=0.8, shuffle=True, random_state=55 )
x1_train, x1_test, y2_train, y2_test = train_test_split(x1,y2,
    train_size=0.8, shuffle=True, random_state=55 )



print(x1_train.shape, x1_test.shape)  # (80, 2) (20, 2)
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

# concatenate 
from tensorflow.python.keras.layers import concatenate, Concatenate   # 사슬 같이 잇다
merge1 = Dense(80, activation='relu', name='mg2')(output1)
merge2 = Dense(50, name='mg3')(merge1)
merge3 = Dense(30, name='mg4')(merge2)
last_output1 = Dense(1, name='last1')(merge3)

merge12 = Dense(80, activation='relu', name='mg12')(output1)
merge13 = Dense(50, name='mg13')(merge12)
merge14 = Dense(30, name='mg14')(merge13)
last_output2 = Dense(1, name='last2')(merge14)
model = Model( inputs=input1, outputs=[last_output1,last_output2] ) # 모델의 정의가 됨

model.summary()


# 맹그러봐

import time
from tensorflow.python.keras.callbacks import EarlyStopping

start_time = time.time()

earlystopping = EarlyStopping(monitor='val_loss', patience=200, mode='min', verbose=1,
                              restore_best_weights=True)

model.compile(loss = 'mse', optimizer = 'adam',
              metrics = ['accuracy'])

a = model.fit(x1_train, [y1_train, y2_train], epochs=2250, batch_size=200,
              validation_split=0.2,
              callbacks= [earlystopping], verbose=1)

end_time = time.time()-start_time
print(a)
print(a.history['val_loss'])

# 4. evaluate , perdict

loss = model.evaluate(x1_test, [y1_test,y2_test])
print('loss : ', loss)
y_predict = model.predict(x1_test)

from sklearn.metrics import r2_score

r2_1 = r2_score(y1_test, y_predict[0])
r2_2 = r2_score(y2_test, y_predict[1])

print('r2score : ', r2_1)
print('r2score : ', r2_2)
print('loss : ', loss)
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




# loss :  [53008.53125, 50195.44140625, 2813.09130859375, 0.0, 0.0]
# r2score :  -41.86370883440396
# r2score :  -1.4022010677516956
# loss :  [53008.53125, 50195.44140625, 2813.09130859375, 0.0, 0.0]
# 걸린시간 :  3.2364485263824463



# loss :  [0.5369672179222107, 0.5250478386878967, 0.011919407173991203, 0.0, 0.0]
# r2score :  0.9995516425520825
# r2score :  0.9999898215852467
# loss :  [0.5369672179222107, 0.5250478386878967, 0.011919407173991203, 0.0, 0.0]
# 걸린시간 :  57.71961307525635
# keras43_ensemble2.py


