import numpy as np
# 이해해라 무조건 
# 컬럼을 늘려도 봐라 
# 
a = np.array(range(1, 11)) # 1 ~ 10
size = 9

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1): # 10 - 9 + 1 = 2
        subset = dataset[i : (i + size)]
        aaa.append(subset)  
    return np.array(aaa)

bbb = split_x(a, size)
print(bbb)  # [[1 2 3 4 5 6 ]]
# print(bbb.shape)  # ( 1, 6)
print('===========================================')
x = bbb[:, :-1]
y = bbb[:, -1]
print('===========================================')
print(x, y) # [[1 2 3 4 5 ]] [ 6 ]
# print(x.shape, y.shape) # ( 1, 5 )    ( 1 , )

