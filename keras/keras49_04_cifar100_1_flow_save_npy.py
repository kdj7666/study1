# 넘파이에서 불러와서 모델구성
# 성능 비교 # 증폭해서 npy지점 
# 48 - 2







# 3.9.7 데이터셋은 인식 python 인식 안됌 
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import time
from keras.datasets import cifar100

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
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

print(x_augmented.shape)   # (400, 28, 28)  카피본 
print(y_augmented.shape)   # (10,)

# 원본 등장

x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 3)

x_augmented = x_augmented.reshape(x_augmented.shape[0],
                                  x_augmented.shape[1],
                                  x_augmented.shape[2], 3)

x_augmented = train_datagen.flow(x_augmented,y_augmented,
                                 batch_size=augment_size,
                                 shuffle=False).next()[0]  # 0번째가 x 1번째가 y로

print(x_augmented)
print(x_augmented.shape) # (400, 32, 32, 3)

# concatenate 사슬처럼 엮다 괄호 2개 필수 제공 나중에 배우지만 찾아서 공부할것
x_train = np.concatenate((x_train, x_augmented)) 
y_train = np.concatenate((y_train, y_augmented))

print(x_train.shape, y_train.shape)  # (50400, 32, 32, 3) (50400, 1)


print('=fd-fd=f-d=fd-f=d-=fwaeisavohanglsa')

# x_train, x_test, y_train, y_test = train_test_split(x,y,
#             train_size=0.7, shuffle=True, random_state=55)
np.save('d:/study_data/_save/_npy/keras49_4_train_x(cifar100).npy', arr=x_train)
np.save('d:/study_data/_save/_npy/keras49_4_train_y(cifar100).npy', arr=y_train)
np.save('d:/study_data/_save/_npy/keras49_4_test_x(cifar100).npy', arr=x_test)
np.save('d:/study_data/_save/_npy/keras49_4_test_y(cifar100).npy', arr=y_test)
# # 넌파이 파일로 저장한다 넌파일수치로 저장이 됨

# x_train = np.load('d:/study_data/_save/_npy/keras49_4_train_x(cifar100).npy')
# y_train = np.load('d:/study_data/_save/_npy/keras49_4_train_y(cifar100).npy')
# x_test = np.load('d:/study_data/_save/_npy/keras49_4_test_x(cifar100).npy')
# y_test = np.load('d:/study_data/_save/_npy/keras49_4_test_y(cifar100).npy')

print(x_train.shape) # (90000, 32, 32, 3)
print(y_train.shape) # (90000, 1)
print(x_test.shape) # (10000, 32, 32, 3)
print(y_test.shape) # (10000,)
