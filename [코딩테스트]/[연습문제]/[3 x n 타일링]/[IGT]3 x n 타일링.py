def solution(n):
    if n%2:
        return 0
    a = b = 1
    for i in range(n//2):
        a,b = 4*a-b, a
    return a%1000000007
