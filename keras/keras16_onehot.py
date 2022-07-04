# [ 과제 ]
# 3가지 원 핫 인코딩 방식을 비교할것 
# 원 핫 인코딩( One Hot Encoding ) 단어 집합의 크기를 벡터 차원으로 만든 뒤,
# 표현하고 싶은 단어의 인덱스에 1의 값을 부여하고 다른 인덱스에는 0을
# 부여하는 방식을 말합니다. 


#1. pandas의 get_dummies 

# get_dummies() 함수 안에 본인이 변환하고자
# 하는 dataframe을 넣어주면 된다 

#2. tensorflow의 to_categorical

# to categorical 은 array 배열이 0부터 시작하기때문에 0을 만들어버림 
# 실제 데이터가 1에서 시작하더라고 0이 생성됨 

#3. sklearn의 OneHotEncoder 

# reshape 와 sparse= False 필수로 쓰임 
# sklearn에서는 onehotencoder를 지원함 

# 미세한 차이를 정리하시오 

