from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import pandas as pd 


file_name_freq += 1
data_aug_gen = ImageDataGenerator(rescale=1./255,
                                  rotation_range=30,
                                  width_shift_range=0.1,
                                  height_shift_range=0.1,
                                  shear_range=0.5,
                                  zoom_range=[0.8,2.0],
                                  horizontal_flip=True,
                                  vetical_flip=True,
                                  fill_mode='nearest')

xy_train = train_datagen.flow_from_directory(
    'd:/pp/',
    target_size=(200, 200),
    batch_size=5,
    class_mode='binary', # 0아니면 1 이기에 binary 2 이상은 categorical
    color_mode='grayscale', # 컬러작업 쓰지않으면 디폴트는 칼라로 인식된다 
    shuffle=True,
)


img = load_img('d:/pp/')
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

i = 0
save_to_dir = fname.split("")[0] + "/"+fname.split("WW")[1]
if not save_to_dir == './imgs_others/train/airplanes':
    for batch in data_aug_gen.flow(x, batch_size=1, save_to_dir=save_to_dir, save_prefix=
                                   'puls_'+str(file_name_freq),
                                   save_format='jpg'):
        i += 1
        if i > 10:
            break




