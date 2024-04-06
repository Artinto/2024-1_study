def solution(n, m):
    x = n*m
    while n%m:
        n,m = m,n%m
    return [m,x/m]
