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

augment_size = 20        # 증강 사이즈 
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

import time
start_time = time.time()
print('시작!!')

x_augmented = train_datagen.flow(x_augmented,y_augmented,
                                 batch_size=augment_size,
                                 save_to_dir='d:/study_data/_temp/',   # 세이브파일은 플로우. 플로우 파일 디렉토리에도 가능하다 
                                 shuffle=False).next()[0]  # 0번째가 x 1번째가 y로

end_time = time.time()-start_time
print(augment_size,'에 걸린시간 : ', round(end_time, 3), '초')


# print(x_augmented)
# print(x_augmented.shape) # 40000 28 28 1

# # concatenate 사슬처럼 엮다 괄호 2개를 제공 필수임 나중에배우지만 찾아서 공부할것
# x_train = np.concatenate((x_train, x_augmented)) 
# y_train = np.concatenate((y_train, y_augmented))

# print(x_train.shape, y_train.shape)  # (100000, 28, 28, 1) (100000,)


