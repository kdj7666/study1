# 3.9.7 데이터셋은 인식 python 인식 안됌 
from tensorflow.keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    # vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    # shear_range=0.7,
    fill_mode='nearest'
)

augment_size = 40000        # 증강 사이즈 
randidx = np.random.randint(x_train.shape[0], size=augment_size)  # np.random.randint = 무작위로 int(정수)값을 넣어준다 
print(randidx) # [20762  8095 11489 ... 58612  1518 52314]
print(x_train.shape) # (60000, 28, 28 )
print(x_train.shape[0]) # 60000
print(np.min(randidx), np.max(randidx))  # 0 59999  랜덤 난수 조절 가능 
print(type(randidx))  # <class 'numpy.ndarray'>

x_augmented = x_train[randidx].copy()     
y_augmented = y_train[randidx].copy()      # x의 값만 뽑을수 없다 y의 값도 같은위치에 같은걸로 만들어줘야한다 

print(x_augmented.shape)   # (40000, 28, 28)  카피본 
print(y_augmented.shape)   # (40000,)

# 원본 등장 

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

x_augmented = x_augmented.reshape(x_augmented.shape[0],
                                  x_augmented.shape[1],
                                  x_augmented.shape[2], 1)

x_augmented = train_datagen.flow(x_augmented,y_augmented,
                                 batch_size=augment_size,
                                 shuffle=False).next()[0]  # 0번째가 x 1번째가 y로

print(x_augmented)
print(x_augmented.shape) # 40000 28 28 1

# concatenate 사슬처럼 엮다 괄호 2개를 제공 필수임 나중에배우지만 찾아서 공부할것
x_train = np.concatenate((x_train, x_augmented)) 
y_train = np.concatenate((y_train, y_augmented))

print(x_train.shape, y_train.shape)  # (100000, 28, 28, 1) (100000,)



'''
print(x_train[0].shape) # (28, 28)
print(x_train[0].reshape(28*28).shape) # (784,) 스칼라가 784 개로 리쉐이프
print(np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1,28,28,1).shape)      # (100, 28, 28, 1)
print(np.zeros(augment_size))
print(np.zeros(augment_size).shape) # (100,)



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

'''