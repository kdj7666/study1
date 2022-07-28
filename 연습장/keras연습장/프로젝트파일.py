from colorsys import yiq_to_rgb
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn import svm
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D ,Flatten, Dense, Dropout
import pandas as pd
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
import tensorflow as tf
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from keras.datasets import mnist, cifar100 , fashion_mnist

train_datagen = ImageDataGenerator(             # 이미지를 수치화. 증폭도 가능. 
    rescale=1./255,                             # 다른 처리 전에 데이터를 곱할 값입니다. 원본 이미지는 0-255의 RGB 계수로 구성되지만 이러한 값은 모델이 처리하기에는 너무 높기 때문에(주어진 일반적인 학습률) 
                                                # 1/255로 스케일링하여 대신 0과 1 사이의 값을 목표로 합니다.
    horizontal_flip=True,                       # 이미지의 절반을 가로로 무작위로 뒤집기 위한 것입니다. 수평 비대칭에 대한 가정이 없을 때 관련이 있습니다
    vertical_flip=True,                         # 이미지의 절반을 가로로 무작위로 뒤집기 위한 것입니다. 수직 비대칭에 대한 가정이 없을 때 관련이 있습니다
    width_shift_range=0.1,                      # width_shift그림을 수직 또는 수평으로 무작위로 변환하는 범위(총 너비 또는 높이의 일부)입니다.
    height_shift_range=-0.1,                    # height_shift 수직 또는 수평으로 무작위로 변환하는 범위(총 너비 또는 높이의 일부)입니다.
    rotation_range=5,                           # 사진을 무작위로 회전할 범위인 도(0-180) 값입니다.
    zoom_range=[0.8,2.0],                       # 내부 사진을 무작위로 확대하기 위한 것입니다
    shear_range=0.7,                            # 무작위로 전단 변환 을 적용하기 위한 것입니다. # 찌그러,기울려 
    fill_mode='nearest'                         # 회전 또는 너비/높이 이동 후에 나타날 수 있는 새로 생성된 픽셀을 채우는 데 사용되는 전략입니다.
)

test_datagen =ImageDataGenerator(               # 평가데이터는 증폭하지 않는다. (수정x)
    rescale=1./255
)

xy = train_datagen.flow_from_directory(
    'D:\pp',
    target_size=(150,150),                       
    batch_size=4000,
    class_mode='categorical',                        
    shuffle=True,
    )                                            

x = xy[0][0]
y = xy[0][1]

print(x.shape,y.shape)  # (5, 100, 100, 3) (5,)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25 )


print(x_train.shape, x_train.shape) #  (1450, 150, 150, 3) (1450, 150, 150, 3)
print(y_test.shape, y_test.shape)   # (550,) (550,)                          


augument_size = 15000                   # 반복횟수
randidx =np.random.randint(x_train.shape[0],size=augument_size)

print(np.min(randidx),np.max(randidx))      # random 함수 적용가능. 
print(type(randidx))            # <class 'numpy.ndarray'>  

x_augumented = x_train[randidx].copy()
y_augumented = y_train[randidx].copy()

print(x_augumented.shape)       # (40000, 150, 150, 1)
print(y_augumented.shape)       # (40000,)

x_augumented = train_datagen.flow(x_augumented, y_augumented, batch_size=augument_size, shuffle=False).next()[0]

# 원본train과 증폭train 합치기
x_train = np.concatenate((x_train, x_augumented))
y_train = np.concatenate((y_train, y_augumented))

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# np.save('d:/study_data/_save/_npy/project_train5_x.npy', arr =x_train)
# np.save('d:/study_data/_save/_npy/project_train5_y.npy', arr =y_train)
# np.save('d:/study_data/_save/_npy/project_test5_x.npy', arr =x_test)
# np.save('d:/study_data/_save/_npy/project_test5_y.npy', arr =y_test)



print('=============================================================================')



import numpy as np
import pandas as pd
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler  
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 

#1. 데이터
# season = np.load('d:/study_data/_save/_npy/personaltest_rain.npy')
# x_train = np.load('d:/study_data/_save/_npy/project_train4_x.npy')
# y_train = np.load('d:/study_data/_save/_npy/project_train4_y.npy')
# x_test = np.load('d:/study_data/_save/_npy/project_test4_x.npy')
# y_test = np.load('d:/study_data/_save/_npy/project_test4_y.npy')

print(x_train.shape)            # (2000, 150, 150, 3)
print(y_train.shape)            # (2000,)
print(x_test.shape)             # (550, 150, 150, 3)
print(y_test.shape)             # (550,)

# x_train = x_train.reshape(2000,450,150)
# x_test = x_test.reshape(550,450,150)


from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten , Dropout,MaxPooling2D,LSTM


#2. 모델 
model = Sequential()
model.add(Conv2D(64,(3,3), input_shape = (150,150,3), padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(256,(3,3), padding='same',activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(128,(3,3), padding='same',activation='relu'))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(3,activation='softmax'))


#3. 컴파일.훈련

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics= ['accuracy'])

earlystopping =EarlyStopping(monitor='loss', patience=20, mode='auto', 
              verbose=1, restore_best_weights = True)     

hist = model.fit(x_train,y_train, epochs=150,validation_split=0.1,verbose=2,batch_size=32,
                 callbacks=[earlystopping])

#4. 예측
accuracy = hist.history['accuracy']
val_accuracy = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss : ',loss[-1])
print('accuracy : ', accuracy[-1])
# loss = model.evaluate(x_test, y_test)
# x_predict = model.predict(x_test)

# y_predict = model.predict(x_test)
# # y_predict = np.argmax(y_predict, axis= 1)
# # y_test = np.argmax(y_test, axis= 1)



# season_predict = model.predict(season)
# y_test = np.argmax(y_test, axis= 1)
# y_predict = np.argmax(season_predict, axis=1) 
# print('predict : ',season_predict)
############################################
loss = model.evaluate(x_test, y_test)
# y_predict = model.predict(season)
print('loss : ', loss)

# y_test = np.argmax(y_test, axis= 1)
# y_predict = np.argmax(y_predict, axis=1) 
# print('predict : ',y_predict)

# acc = accuracy_score(y_test,y_predict) 
# print('acc : ', acc) 

