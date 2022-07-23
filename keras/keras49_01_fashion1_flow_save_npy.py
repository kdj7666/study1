# 48 - 3 ㅂㅂ
# 3.9.7 데이터셋은 인식 python 인식 안됌 
from click import argument
from tensorflow.keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import time

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
train_datagen2 = ImageDataGenerator(
    rescale=1./255,)


augment_size = 40000        # 증강 사이즈 

# batch_size=100000

randidx = np.random.randint(x_train.shape[0], size=augment_size)  # np.random.randint = 무작위로 int(정수)값을 넣어준다 

x_augmented = x_train[randidx].copy()     
y_augmented = y_train[randidx].copy()      # x의 값만 뽑을수 없다 y의 값도 같은위치에 같은걸로 만들어줘야한다 

print(x_augmented.shape)   # (40000, 28, 28)  카피본 
print(y_augmented.shape)   # (40000,)
x_train = x_train.reshape(60000, 28, 28, 1)
x_augmented = x_augmented.reshape(x_augmented.shape[0],
                                  x_augmented.shape[1],
                                  x_augmented.shape[2], 1)

xy_ab = x_train.reshape(x_train.shape[0],
                        x_train.shape[1],
                        x_train.shape[2], 1)

x_train1 = train_datagen.flow(x_augmented,y_augmented,
                                 batch_size=augment_size, shuffle=False)

x_ab = np.concatenate((x_train, x_augmented)) 
y_ab = np.concatenate((y_train, y_augmented))

xy_ab2 = train_datagen2.flow(x_ab, y_ab,
                              batch_size=augment_size, shuffle=False)


print(x_augmented)
print(x_augmented.shape) # 40000 28 28 1

# concatenate 사슬처럼 엮다 괄호 2개를 제공 필수임 나중에배우지만 찾아서 공부할것
# x_train = np.concatenate((x_train, x_augmented)) 
# y_train = np.concatenate((y_train, y_augmented))
print('======================================================')
print(x_train.shape, y_train.shape)  # (60000, 28, 28, 1) (60000,)



# x_train, x_test, y_train, y_test = train_test_split(x,y,
#             train_size=0.7, shuffle=True, random_state=55)
np.save('d:/study_data/_save/_npy/keras49_1_train_x(fashion_mnist).npy', arr=x_train)
np.save('d:/study_data/_save/_npy/keras49_1_train_y(fashion_mnist).npy', arr=y_train)
np.save('d:/study_data/_save/_npy/keras49_1_test_x(fashion_mnist).npy', arr=x_test)
np.save('d:/study_data/_save/_npy/keras49_1_test_y(fashion_mnist).npy', arr=y_test)
# # 넌파이 파일로 저장한다 넌파일수치로 저장이 됨

# x_train = np.load('d:/study_data/_save/_npy/keras49_1_train_x(fashion_mnist).npy')
# y_train = np.load('d:/study_data/_save/_npy/keras49_1_train_y(fashion_mnist).npy')
# x_test = np.load('d:/study_data/_save/_npy/keras49_1_test_x(fashion_mnist).npy')
# y_test = np.load('d:/study_data/_save/_npy/keras49_1_test_y(fashion_mnist).npy')

# print(x_train.shape) # (10, 150, 150, 1)
# print(y_train.shape) # (10,)
# print(x_test.shape) # (10, 150, 150, 1)
# print(y_test.shape) # (10,)
