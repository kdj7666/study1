# 3.9.7 데이터셋은 인식 python 인식 안됌 
from tensorflow.keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    shear_range=0.7,
    fill_mode='nearest'
)

augument_size = 100        # 증강 사이즈 

print(x_train[0].shape) # (28, 28)
print(x_train[0].reshape(28*28).shape) # (784,) 스칼라가 784 개로 리쉐이프
print(np.tile(x_train[0].reshape(28*28), augument_size).reshape(-1,28,28,1).shape)      # (100, 28, 28, 1)
print(np.zeros(augument_size))
print(np.zeros(augument_size).shape) # (100,) # np.zeros 기존 배열에 0으로 채워진 모양과 유형의 배열을 반환시킨다
                                     # 0~99이지만 들어가있는 배열은 0~99 실제는 0~00 
x_data = train_datagen.flow(
    np.tile(x_train[0].reshape(28*28), augument_size).reshape(-1,28,28,1),   # x
    np.zeros(augument_size),                                                 # y
    batch_size=augument_size,                                                # 배치사이즈에 넣는 엑스와이는 ㅍ르로우에 이다
    shuffle=True,
) # .next() # [] 하나를 제거시킨다 인식이 가능하다 


#### .next() 사용
# print(x_data)  # 데이터 형태 확인
# print(x_data[0])  # 배치사이즈 위치  x y 값이 같이 있음 
# print(x_data[0].shape) # (100, 28, 28, 1)
# print(x_data[1].shape) # (100,)

#### .next() 미사용
print(x_data)
print(x_data[0])  # 배치사이즈 위치  x y 값이 같이 있음 
print(x_data[0][0].shape) # (100, 28, 28, 1)  x
print(x_data[0][1].shape) # (100,)  y


import matplotlib.pyplot as plt
plt.figure(figsize=(7,7))
for i in range(49):
    plt.subplot(7, 7, i+1)
    plt.axis('off')
    plt.imshow(x_data[0][0][i], cmap='gray') # .next() 를 안쓸때 대괄호 하나를 더 넣어준다 
    # plt.imshow(x_data[1], cmap='gray')
plt.show()

