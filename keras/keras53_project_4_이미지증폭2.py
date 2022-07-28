import time
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator( # 이미지 데이터를 수치화 
    rescale=1./255,
    horizontal_flip=True,   # 수평 반전
    vertical_flip=True,     # 수직 반전
    width_shift_range=0.1,  # 가로넒이을 0.1만 옮길수있다
    height_shift_range=0.1,  # 세로넒이를 0.1만 옮길수있다            shitf 옮기다 
    rotation_range=15,       # 회전은 5만 할수있다 
    zoom_range=1.2,     # 확대
    shear_range=0.7,    # 깎다
    fill_mode='nearest'  # 채우다 
)  # 트레인 데이터를 이렇게 수치화 할꺼야 ( 준비 ) 여기까지 안엮인것
test_datagen = ImageDataGenerator(
    rescale=1./255
)

# xy_train 을 폴더에서 가져오겠다 폴더를 directory 
train = train_datagen.flow_from_directory(
    'd:/pp/',
    target_size=(200, 200),
    batch_size=5,
    class_mode='categorical', # 0아니면 1 이기에 binary 2 이상은 categorical
    # color_mode='grayscale', # 컬러작업 쓰지않으면 디폴트는 칼라로 인식된다 
    shuffle=True,
)   # Found 405 images belonging to 3 classes.

test = test_datagen.flow_from_directory(
    'd:/pp2/',
    target_size=(200, 200),
    batch_size=5,
    class_mode='categorical', # 0아니면 1 이기에   [ ad , normal ] binary 2 이상은 categorical
    # color_mode='grayscale',  # 컬러 작업 
    shuffle=True,
)   # Found 3 images belonging to 1 classes.


datagen = ImageDataGenerator(rotation_range = 90,
                             width_shift_range=0.4,
                             height_shift_range=0.4,
                             vertical_flip =True,
                             horizontal_flip =True)

for x_data, t_data in train:
    print(x_data.shape)  # (20, 150, 150, 3)
    print(type(x_data))  # <class 'numpy.ndarray'>
    print(t_data)

idx = 0
fig = plt.figure(figsize=(10, 10))
axs = []
for batch in datagen.flow(train , batch_size=1): # 여기서 batch는 x가 됨
    axs.append(flg.add_subplot(5, 4, idx+1))
    axs[idx].imshow(image.array_to_img(batch[0]))
    idx += 1
    if idx%20 == 0:
        break
fig.tight_layout()
plt.show()













'''
img = train_datagen.flow_from_directory(
    'd:/pp/',)

x = img_to_array(img)
x = x.reshape((1, ) + x.shape)


































print(xy_train[0][0]) # (5, 150, 150, 3)
print(xy_train[0][1])

print(xy_train[0][0].shape, xy_train[0][1].shape)  # (80, 200, 200, 1) (80,)

x_train = xy_train[0][0]
y_train = xy_train[0][1]
# x_test = xy_test[0][0]
# y_test = xy_test[0][1]







print(type(xy_train))
print(type(xy_train[0]))
print(type(xy_train[0][0]))
print(type(xy_train[0][1]))

augment_size = 480000        # 증강 사이즈 
randidx = np.random.randint(x_train.shape[0], size=augment_size)  # np.random.randint = 무작위로 int(정수)값을 넣어준다 


print(y_train.shape) # 5, 


#2. model
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(64, (2,2), input_shape=(200, 200, 1), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. compile, epochs

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# model.fit(xy_train[0][0], xy_train[0][1]) # 허나 배치를 최대로 잡으면 이것도 가능하다

start_time = time.time()

hist = model.fit_generator(xy_train, epochs=10, steps_per_epoch=64,    
                         # 스텝 펄 에포 ( 통상적으로 batch= 160/5 = 32)  # 훈련 배치 사이즈가 32가 넘어서도 돌아가긴한다 추가적 환경 제공 가능
                    validation_data=xy_test,                    # 발리데이션 범주를 테스트로
                    validation_steps=5)                         # 발리데이션 스텝 : 한 epoch 종료 시 마다 검증할 때 사용되는 검증 스텝 수를 지정합니다

end_time = time.time()-start_time

#4. evluate, predict
acc = hist.history['accuracy']
# val_accuracy = hist.history['val_accuracy']
loss = hist.history['loss']
# val_loss = hist.history['val_loss']

print('loss : ', loss[-1]) # 마지막 괄호로 마지막 1개만 보겠다
# print('val_loss : ', val_loss[-1])
# print('val_accuracy : ', val_accuracy[-1])
print('accuracy : ', acc[-1])
print('걸린시간 : ', end_time)

'''
