# 함수 
def max( a, b):
    return a > b

print(max(20, 10))

def SayHello(name, say = '반갑습니다'):
    print("{0}님 {1}" .format(name, say))

SayHello("강동진")
SayHello("고객", "안녕하세요")

# 클래스 

class Person:
    def __init__(self, name, age): #생성자
        self.name = name
        self.age = age
        
    def func(self, str = "동해번쩍 서해번쩍"):
        print("{0} : {1}, {2}" .format(self.name, self.age, str))

p = Person("홍길동" , 25)
p.func("호형호제")


'''
class Person:
    def __init__(self,name,wallet):   # 생성자 
        self.name = name
        self.wallet = wallet
    def money(self):
        print(self.wallet)
bill = Person('bill','10000')
bill.money()

'''