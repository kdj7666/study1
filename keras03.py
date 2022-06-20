# 과제 파이썬 리스트 알아오기

list1 = [2, 5, 7, 9, 10]
print(list1[0])
print(list1[3])
print(list1[2]+list1[-1]) # -1 은 뒤에서부터 시작

# 리스트 요소 연산자 숫자는 0부터 시작 
a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print(a[2])
print(a[2][1])

# 문자열일때
b = [["문, 자, 열"], ['세', '글', '자']]
print(b[0])
print(b[1])
print(b[1][0])

# 리스트란 숫자형 문자열 구을링형 리스트 한꺼번에 모아서
# 저장하여 값을 변경하거나 수행하는 것을 리스트 

# 리스트의 순서는 인덱스

num = [1,2,3,4,5]
# 리스트 출력하기
print(num)
print(num[4])
food = ['첵스초코', '도시락', '소보루빵', '팔도 비빔면']
# 문자열 리스트 출력하기
print(food)
print(food[3])

# 리스트의 특정 구간을 자르는 건 슬라이싱
print(num[0:2])

print(food[-3])

# 리스트 랜덤으로 뽑을수 있음 
students = ['메이충리엘', 'Kelly', '남재우'
            , '김민수', '공동동']

students
for stu in students:
    print(stu)

import random
print(random.choice(students))
