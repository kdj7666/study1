# 리스트는 항상 대괄호 
# 행렬은 아래방향을 우선으로 채워짐
# 특성 컬럼 열 피쳐
# 스칼라 0차원 벡터는 1차원 매트리스 행렬이 2차원 텐서는 3차원 '''' 4차원 텐서 5차원 텐서
# 피쳐의 갯수는 동일 렬은 갯수가 같다 ex) [1,2,3],[4,3] (x)
# 행 무시 열 우선 

# 리스트의 순서는 인덱스

num = [1,2,3,4,5]
# 리스트 출력하기
print(num)
print(num[4])
food = ['첵스초코', '도시락', '소보루빵', '팔도 비빔면']
# 문자열 리스트 출력하기
print(food)
print(food[2])

# 리스트의 특정 구간을 자르는 건 슬라이싱
print(num[0:2])

print(food[-2])


em_list = [1, '한글', [1,2,3]]