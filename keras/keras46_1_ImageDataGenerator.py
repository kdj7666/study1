import numpy as np
from keras.preprocessing.image import ImageDataGenerator


# from tensorflow.python.keras import image
# from tensorflow.python.keras import ImageDataGenerator
# 대문자는 90%이상 클래스
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
    'd:/_data/image/brain/train/',
    target_size=(200, 200),
    batch_size=5,
    class_mode='binary', # 0아니면 1 이기에 binary 2 이상은 categorical
    color_mode='grayscale', # 컬러작업 쓰지않으면 디폴트는 칼라로 인식된다 
    shuffle=True,
)   # Found 160 image belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    'd:/_data/image/brain/test/',
    target_size=(200, 200),
    batch_size=5,
    class_mode='binary', # 0아니면 1 이기에   [ ad , normal ] binary 2 이상은 categorical
    color_mode='grayscale',  # 컬러 작업 
    shuffle=True,
)   # Found 120 image belonging to 2 classes.

print(xy_train)
# <keras.preprocessing.image.DirectoryIterator object at 0x000001BB8A67CD90>

# from sklearn.datasets import load_boston

# datasets = load_boston()
# print(datasets)

print(xy_train[0])  # xy값이 같이 포함되어있다 y가 5개가 포함되어있다 배치사이즈 5 
# ValueError: Asked to retrieve element 33, but the Sequence has length 32 - 총 160개의 데이터가 5개식 짤려 32개가 있는데
# 33개는 되지 않는다 32개도 본인 포함이기에 31까지만 가능하다   마지막 배치는 31

print(xy_train[0][0]) # (5, 150, 150, 3)
print(xy_train[0][1]) # 첫번째 가로는 잘라진 구간의 번호를 표출  
                       # 두번째 가로는 x와 y의 값을 표현 ( 0번째는 x , 1번째는 y )

# print(xy_train[31][2]) # 0과 1만 존재하기에 2는 에러가 나온다 


print(xy_train[0][0].shape, xy_train[0][1].shape)  # (80, 200, 200, 1) (80,)

                             # 이미지 데이터의 쉐이프는 type 로 확인한다 
print(type(xy_train))        # <class 'keras.preprocessing.image.DirectoryIterator'>  iteratior 반복자  for문 
print(type(xy_train[0]))     # <class 'tuple'> tuple : 생성 삭제 수정이 불가능 ( x tuple y tuple 두개 들어가있다 ) 
print(type(xy_train[0][0]))  # <class 'numpy.ndarray'>  # x의 값을 보려 [0]
print(type(xy_train[0][1]))  # <class 'numpy.ndarray'>  # y의 값을 보려 [1]

