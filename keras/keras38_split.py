import numpy as np
# 이해해라 무조건 
# 컬럼을 늘려도 봐라 
# range 함수는 마지막 숫자 -1 
a = np.array(range(1, 11)) # 1 ~ 10
size = 9

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1): # 10 - 9 + 1 = 2
        subset = dataset[i : (i + size)]
        aaa.append(subset)  
    return np.array(aaa)

f = np.array(range(31, 41)) # 31 32 33 34 35 36 37 38 39 40 
size = 3 

def split_g(dataset, size):
    rrr = []
    for l in range(len(dataset) - size + 1):
        subset = dataset[l : (l + size)]
        rrr.append(subset)
    return np.array(rrr)


bbb = split_x(a, size)
print(bbb)  # [[1 2 3 4 5 6 ]]
# print(bbb.shape)  # ( 1, 6)
print('===========================================')
x = bbb[:, :-1]  # x 의 데이터를 범위 까지 뽑되 마지막 범위는 제외     + 는 앞 - 는 뒤 ( 뒤에서 첫번째 )
y = bbb[:, -1]   # y의 데이터를 범위 까지 뽑되 /  마지막 y를 가지고와  ( 뒤에서 첫번째 )
print('===========================================')
print(x, y) # [[1 2 3 4 5 ]] [ 6 ]
# print(x.shape, y.shape) # ( 1, 5 )    ( 1 , )
print('===========================================')

qqq = split_g(f, size)
print(qqq)

print('===========================================')
x = qqq[:, :-1]   # x 의 데이터를 범위 까지 뽑되 마지막 범위는 제외     + 는 앞 - 는 뒤 ( 뒤에서 첫번째 )
y = qqq[:, -1]    # y의 데이터를 범위 까지 뽑되 /  마지막 y를 가지고와  ( 뒤에서 첫번째 )
print('===========================================')

print(x,y)

