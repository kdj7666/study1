import numpy as np
from keras.preprocessing.image import ImageDataGenerator

# 1. 데이터
train_datagen = ImageDataGenerator(
    # rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=5,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest'
    )

scale_datagen = ImageDataGenerator(rescale=1./255)

xy_train = scale_datagen.flow_from_directory(
    'd:/study_data/_data/cat_dog/training_set/',
    target_size=(150, 150),
    batch_size=500,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True
)

xy_test = scale_datagen.flow_from_directory(
    'd:/study_data/_data/cat_dog/test_set/',
    target_size=(150, 150),
    batch_size=500,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True
) # Found 120 images belonging to 2 classes.

x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]

# 증폭 사이즈만큼 난수 뽑아서
augument_size = 500
randidx = np.random.randint(x_train.shape[0], size=augument_size)
# 각각 인덱스에 난수 넣고 돌려가면서 이미지 저장
x_augument = x_train[randidx].copy()
y_augument = y_train[randidx].copy()

# x 시리즈 전부 리쉐입
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_augument = x_augument.reshape(x_augument.shape[0], x_augument.shape[1], x_augument.shape[2], 1)

# x 증폭 데이터 담기
x_augument = train_datagen.flow(x_augument, y_augument, batch_size=augument_size, shuffle=False).next()[0]

x_train = scale_datagen.flow(x_train, y_train, batch_size=augument_size, shuffle=False).next()[0]

# 원본train과 증폭train 합치기
x_train = np.concatenate((x_train, x_augument))
y_train = np.concatenate((y_train, y_augument))

np.save('d:/study_data/_save/_npy/keras49_6_train_x(cat_dog).npy', arr =x_train)
np.save('d:/study_data/_save/_npy/keras49_6_train_y(cat_dog).npy', arr =y_train)
np.save('d:/study_data/_save/_npy/keras49_6_test_x(cat_dog).npy', arr =x_test)
np.save('d:/study_data/_save/_npy/keras49_6_test_y(cat_dog).npy', arr =y_test)

