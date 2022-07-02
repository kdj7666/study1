'''
def addfunction(a, b): 
    return a + b
def add(a, b): 
    return a + b
        
c = add(2, 3) #함수 호출
print(c)    
def add(a, b): #a,b는 매개변수
    return a + b
        
c = add(2, 3) #2,3은 인수
print(c)   
'''
class Person:
    def __init__(self, name, age): #생성자
        self.name = name
        self.age = age
        
    def func(self, str = "동해번쩍 서해번쩍"):
        print("{0} : {1}, {2}" .format(self.name, self.age, str))

p = Person("홍길동" , 25)
p.func("호형호제")
