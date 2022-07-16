from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D # 이미지는 2차원 2d

model = Sequential()
# model.add(Dense(units=10, input_shape=(3,)))   # 이 줄은 모델때문이 아니고 units 를 위해서 만든것 # ( batch_size, input_dim) 와꾸
# model.summary()
# ( input_dim + bias ) * units = summary Param 갯수 ( Dense 모델 ) 
# ( kernal_size * channels + bias ) * filters = summary Param 갯수 ( CNN 모델 )


    #  출력 ( 4, 4, 10 ) 
model.add(Conv2D(filters=64, kernel_size=(3,3),    # filters = 10  이거싱 아웃풋 갯수 늘릴수있으며 컴퓨터가 좋아야한다 filters 의 아웃풋이 channels 의 인풋 
                 padding='same',
                 input_shape=(28, 28, 1)))   # ( batch_size, rows, columns, channels )  6 6 10
model.add(MaxPooling2D()) # ( 반 나눈다 )
model.add(Conv2D(32, (2,2), 
                 padding='valid',
                 activation='relu'))  # ( cnn 모델도 한정이 가능하기에 activation 가능 ) # 출력 : ( 3, 3, 7 )
model.add(Flatten()) # 3, 3, 7  을 받았다   # ( N, 63)
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

