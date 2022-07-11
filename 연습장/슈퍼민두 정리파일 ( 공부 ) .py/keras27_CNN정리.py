# 이미지를 잘라서 행렬연산을 해준 후 그 수치를 따로 빼줌 

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D # 이미지 작업은 이거
from tensorflow.python.keras.layers import Flatten # 쫙 펴주는 애
from tensorflow.python.keras.layers import MaxPooling2D

model = Sequential()
# model.add(Dense(units=10, input_shape=(5, 5, 1))) <- 이렇게 해줘도 아래의 첫번째 Conv2D와 같음
# model.add(Dense(units=10, input_shape=(3, ))) # input_shape=(batzh_size, input_dim)
# model.summary()  # (input_dim + bias) * units = summary Param 갯수 (Dense레이어 모델)

model.add(Conv2D(filters=10, kernel_size=(2,2), # 출력 (N, 4, 4, 10) # kernel_size=(이미지를 자르는 규격) 
                 input_shape=(5, 5, 1))) # (N, 5, 5, 1) -> (batch_size, rows, colums, channels) -> (이미지 개수, 행, 열, 흑백 or 컬러 정의) ### 이 input_shape는 모델에서 첫 레이어일 떄만 정의하면 됨
# 다음 레이어의 input_shape으로 전달 될 때는 (n, 4, 4, 10) -> (N, (행-커널사이즈의 행)+1/1, (열-커널사이즈의 열)+1/1, 필터값)
# model.summary()  # (kernel_size * channels + bias) * filters = summary Param 갯수 (CNN 모델)

# model.add(Conv2D(7, (2,2), activation='relu')) 

model.add(MaxPooling2D()) # dropout과 비슷한 개념인듯? conv2d가 kernel을 이용해서 중첩시키며 특성을 추출해나간다면 maxpoolig은 픽셀을 묶어서 그중에 가장 큰값만 뺌
                          # MaxPooling은 값을 반으로 계속 줄여나감. default = 2,2 (4픽셀당 1개값)

model.add(Conv2D(7, (2,2), 
                 padding='valid', # 이게 padding의 디폴트 값. 'same' 해주면 커널 사이즈에 상관없이 원래 shape 그대로 감 (보통 0을 씌움)
                 activation='relu')) # 출력 (N, 3, 3, 7)

# CNN 모델의 출력은 (10, ) or (10, 1) 이러한 2차원 형태로 출력 돼야함
# 따라서 Dense 형태로 바꿔줘야 함 (4차원의 형태를 2차원의 형태로 바꿔줘야 함)
# 어느정도 작업을 하다가 중간에 쫙 펴줌 (reshape 개념) 그 뒤로 Dense 레이어로 진행

model.add(Flatten()) # <- 쫙 펴주기 위해선 얘를 써줘야 함. 위에서 넘겨주는 값을 일렬로 쭉 나열해서 -> (N, 63) 그래서 이렇게 바뀜

model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax')) # 
model.summary()

# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# conv2d (Conv2D)              (None, 4, 4, 10)          50        
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 3, 3, 7)           287        # Param = (커널사이즈곱 * 채널의 갯수 + bias) * 아웃풋 노드의 개수(필터)       
# _________________________________________________________________
# flatten (Flatten)            (None, 63)                0
# _________________________________________________________________
# dense (Dense)                (None, 32)                2048
# _________________________________________________________________
# dense_1 (Dense)              (None, 32)                1056
# _________________________________________________________________
# dense_2 (Dense)              (None, 10)                330
# =================================================================
# Total params: 3,771
# Trainable params: 3,771
# Non-trainable params: 0
# _________________________________________________________________


# tf.keras.layers.Dense(
#     units, -> 아웃풋 노드의 개수
#     activation=None,
#     use_bias=True,
#     kernel_initializer="glorot_uniform",
#     bias_initializer="zeros",
#     kernel_regularizer=None,
#     bias_regularizer=None,
#     activity_regularizer=None,
#     kernel_constraint=None,
#     bias_constraint=None,
#     **kwargs
# )
