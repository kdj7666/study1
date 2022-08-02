import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import time
from sklearn.model_selection import train_test_split

# 1. data

train_datagen = ImageDataGenerator( # 이미지 데이터를 수치화 
    rescale=1./255,
    horizontal_flip=True,   # 수평 반전
    vertical_flip=True,     # 수직 반전
    width_shift_range=0.1,  # 가로넒이을 0.1만 옮길수있다
    height_shift_range=0.1,  # 세로넒이를 0.1만 옮길수있다            shitf 옮기다 
    rotation_range=5,       # 회전은 5만 할수있다 
    zoom_range=1.2,     # 확대
    shear_range=0.7,    # 깎다
    fill_mode='nearest')  # 채우다 

test_datagen = ImageDataGenerator(
    rescale=1./255)

# xy_train 을 폴더에서 가져오겠다 폴더를 directory 
xy_train = train_datagen.flow_from_directory(
    'd:/pp',
    target_size=(150, 150),
    batch_size=10000,
    class_mode='categorical', # 0아니면 1 이기에 binary 2 이상은 categorical   / ? classes 의 갯수로 덩어리를 쪼갤수 있다
    # color_mode='grayscale', # 컬러작업 쓰지않으면 디폴트는 칼라로 인식된다 
    shuffle=True,)

print(xy_train[0][0].shape)

# Found 1000 images belonging to 5 classes.

x = xy_train[0][0]
y = xy_train[0][1]
print(x,x.shape)  # (1000, 150, 150, 3)
print(y,y.shape)  # (1000, 150, 150, 3)


k = test_datagen.flow_from_directory(
    'D:/pp2/',
    target_size=(150, 150),
    batch_size=4,
    class_mode='binary', # 0아니면 1 이기에 binary 2 이상은 categorical   / ? classes 의 갯수로 덩어리를 쪼갤수 있다
    # color_mode='grayscale', # 컬러작업 쓰지않으면 디폴트는 칼라로 인식된다 
    shuffle=True,)
# Found 1 images belonging to 1 classes.
# Found 5 images belonging to 1 classes.

x_train, x_test, y_train, y_test = train_test_split(x,y,
            train_size=0.7, shuffle=False)#, random_state=55)

augument_size = 5000                   # 반복횟수
randidx =np.random.randint(x_train.shape[0],size=augument_size)
 
print(np.min(randidx),np.max(randidx))      # random 함수 적용가능. 
print(type(randidx))            # <class 'numpy.ndarray'>  

x_augumented = x_train[randidx].copy()
y_augumented = y_train[randidx].copy()

print(x_augumented.shape)       # (15000, 150, 150, 1)
print(y_augumented.shape)       # (15000,)

x_augumented = train_datagen.flow(x_augumented, y_augumented, batch_size=augument_size, shuffle=False).next()[0]

# 원본train과 증폭train 합치기
x_train = np.concatenate((x_train, x_augumented))
y_train = np.concatenate((y_train, y_augumented))


np.save('d:/study_data/_save/_npy/keras53-13_train_x.npy', arr=x_train)
np.save('d:/study_data/_save/_npy/keras53-13_train_y.npy', arr=y_train)
np.save('d:/study_data/_save/_npy/keras53-13_test_x.npy', arr=x_test)
np.save('d:/study_data/_save/_npy/keras53-13_test_y.npy', arr=y_test)
np.save('d:/study_data/_save/_npy/keras53-13_test_k.npy', arr=k[0][0])
print('================================================================')


