import math
def solution(n, m):
    def lcm(n,m):
        return n * m // math.gcd(n,m) # 최소공배수 = n * (m을 n과m의 최대공약수로 나눈 값) ex) 12 x (18 // 6) = 36
    return [math.gcd(n,m), lcm(n,m)] # gcd : 최대공약수 구하는 라이브러리

'''
import math
def solution(n, m):
    return [math.gcd(n,m), math.lcm(n,m)] # lcm : 최소공배수 구하는 라이브러리
''' 
