import numpy as np 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator( # 이미지 데이터를 수치화 
    rescale=1./255,
    horizontal_flip=True,   # 수평 반전
    vertical_flip=True,     # 수직 반전
    width_shift_range=0.1,  # 가로넒이을 0.1만 옮길수있다
    height_shift_range=0.1,  # 세로넒이를 0.1만 옮길수있다            shitf 옮기다 
    rotation_range=5,       # 회전은 5만 할수있다 
    zoom_range=1.2,     # 확대
    shear_range=0.7,    # 깎다
    fill_mode='nearest'  # 채우다 
)  # 트레인 데이터를 이렇게 수치화 할꺼야 ( 준비 ) 여기까지 안엮인것
test_datagen = ImageDataGenerator(
    rescale=1./255
)

# xy_train 을 폴더에서 가져오겠다 폴더를 directory 
xy_train = train_datagen.flow_from_directory(
    'd:/pp/',
    target_size=(200, 200),
    batch_size=5,
    class_mode='binary', # 0아니면 1 이기에 binary 2 이상은 categorical
    color_mode='grayscale', # 컬러작업 쓰지않으면 디폴트는 칼라로 인식된다 
    shuffle=True,
)   # Found 405 images belonging to 3 classes.

xy_test = test_datagen.flow_from_directory(
    'd:/pp2/',
    target_size=(200, 200),
    batch_size=5,
    class_mode='binary', # 0아니면 1 이기에   [ ad , normal ] binary 2 이상은 categorical
    color_mode='grayscale',  # 컬러 작업 
    shuffle=True,
)   # Found 3 images belonging to 1 classes.

