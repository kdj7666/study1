import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import time
from sklearn.model_selection import train_test_split

# 1. data

train_datagen = ImageDataGenerator( # 이미지 데이터를 수치화 
    rescale=1./255)
    # horizontal_flip=True,   # 수평 반전
    # vertical_flip=True,     # 수직 반전
    # width_shift_range=0.1,  # 가로넒이을 0.1만 옮길수있다
    # height_shift_range=0.1,  # 세로넒이를 0.1만 옮길수있다            shitf 옮기다 
    # rotation_range=5,       # 회전은 5만 할수있다 
    # zoom_range=1.2,     # 확대
    # shear_range=0.7,    # 깎다
    # fill_mode='nearest')  # 채우다 

# test_datagen = ImageDataGenerator(
#     rescale=1./255)

# xy_train 을 폴더에서 가져오겠다 폴더를 directory 
xy_train = train_datagen.flow_from_directory(
    'd:/study_data/_data/horse-or-human/',
    target_size=(50, 50),
    batch_size=1027,
    class_mode='binary', # 0아니면 1 이기에 binary 2 이상은 categorical classes
    # color_mode='', # 컬러작업 쓰지않으면 디폴트는 칼라로 인식된다 
    shuffle=True,)


print(xy_train)
# Found 1027 images belonging to 2 classes.

# <keras.preprocessing.image.DirectoryIterator object at 0x000001BB8A67CD90>

# from sklearn.datasets import load_boston

# datasets = load_boston()
# print(datasets)


x = xy_train[0][0]
print(x,x.shape)  # (1027, 50, 50, 3)

y = xy_train[0][1]
print(y,y.shape) # (1027,)


x_train, x_test, y_train, y_test = train_test_split(x,y,
            train_size=0.7, shuffle=True, random_state=55)
np.save('d:/study_data/_save/_npy/keras47_2_train_x(horse_or_human).npy', arr=x_train[0][0])
np.save('d:/study_data/_save/_npy/keras47_2_train_y(horse_or_human).npy', arr=y_train[0][1])
np.save('d:/study_data/_save/_npy/keras47_2_test_x(horse_or_human).npy', arr=x_test[0][0])
np.save('d:/study_data/_save/_npy/keras47_2_test_y(horse_or_human).npy', arr=x_test[0][1])
# # 넌파이 파일로 저장한다 넌파일수치로 저장이 됨

# x_train = np.load('d:/study_data/_save/_npy/keras47_2_train_x(horse_or_human).npy')
# y_train = np.load('d:/study_data/_save/_npy/keras47_2_train_y(horse_or_human).npy')
# x_test = np.load('d:/study_data/_save/_npy/keras47_2_test_x(horse_or_human).npy')
# y_test = np.load('d:/study_data/_save/_npy/keras47_2_test_y(horse_or_human).npy')

# print(x_train.shape) # (10, 150, 150, 1)
# print(y_train.shape) # (10,)
# print(x_test.shape) # (10, 150, 150, 1)
# print(y_test.shape) # (10,)
















# print(xy_train[0])  # xy값이 같이 포함되어있다 y가 5개가 포함되어있다 배치사이즈 5 
# # ValueError: Asked to retrieve element 33, but the Sequence has length 32 - 총 160개의 데이터가 5개식 짤려 32개가 있는데
# # 33개는 되지 않는다 32개도 본인 포함이기에 31까지만 가능하다   마지막 배치는 31

# print(xy_train[0][0]) # (5, 150, 150, 3)
# print(xy_train[0][1]) # 첫번째 가로는 잘라진 구간의 번호를 표출  
#                        # 두번째 가로는 x와 y의 값을 표현 ( 0번째는 x , 1번째는 y )

# # print(xy_train[31][2]) # 0과 1만 존재하기에 2는 에러가 나온다 


# print(xy_train[0][0].shape, xy_train[0][1].shape)  # (10, 150, 150, 1) (10,)

#                              # 이미지 데이터의 쉐이프는 type 로 확인한다 
# print(type(xy_train))        # <class 'keras.preprocessing.image.DirectoryIterator'>  iteratior 반복자  for문 
# print(type(xy_train[0]))     # <class 'tuple'> tuple : 생성 삭제 수정이 불가능 ( x tuple y tuple 두개 들어가있다 ) 
# print(type(xy_train[0][0]))  # <class 'numpy.ndarray'>  # x의 값을 보려 [0]
# print(type(xy_train[0][1]))  # <class 'numpy.ndarray'>  # y의 값을 보려 [1]

# # 현재 ( 10, 150, 150, 1 짜리 데이터가 32덩어리 0~31 )





#2. model
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(64, (2,2), input_shape=(50, 50, 3), activation='relu'))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. compile, epochs

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

start_time = time.time()

hist = model.fit(xy_train[0][0], xy_train[0][1], epochs=100, batch_size=15,
          validation_split=0.1, verbose=1) # 허나 배치를 최대로 잡으면 이것도 가능하다


# hist = model.fit_generator(xy_train, epochs=10, steps_per_epoch=32,    
#                          # 스텝 펄 에포 ( 통상적으로 batch= 160/5 = 32)  # 훈련 배치 사이즈가 32가 넘어서도 돌아가긴한다 추가적 환경 제공 가능
#                     validation_data=xy_test,                    # 발리데이션 범주를 테스트로
#                     validation_steps=2, verbose=1)                         # 발리데이션 스텝 : 한 epoch 종료 시 마다 검증할 때 사용되는 검증 스텝 수를 지정합니다

end_time = time.time()-start_time

#4. evluate, predict
acc = hist.history['accuracy']
val_accuracy = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss : ', loss[-1]) # 마지막 괄호로 마지막 1개만 보겠다
print('val_loss : ', val_loss[-1])
print('val_accuracy : ', val_accuracy[-1])
print('accuracy : ', acc[-1])
print('걸린시간 : ', end_time)



# 그림그리기 

import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

font_path = 'C:\Windows\Fonts\malgun.ttf'
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.plot(hist.history['val_accuracy'], marker='.', c='pink', label='val_accuracy')
plt.plot(hist.history['accuracy'], marker='.', c='green', label='accuracy')
plt.grid()
plt.title('loss & val_loss')    
plt.title('로스값과 검증로스값')    
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')   # 우측상단에 라벨표시
plt.legend()   # 자동으로 빈 공간에 라벨표시
plt.show()


# loss :  0.6771479249000549
# val_loss :  0.6616235375404358
# val_accuracy :  0.6499999761581421
# accuracy :  0.5874999761581421
# 걸린시간 :  478.4866187572479

