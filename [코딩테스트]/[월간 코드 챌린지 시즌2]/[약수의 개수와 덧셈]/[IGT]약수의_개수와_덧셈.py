def solution(left, right):
    return sum(n*(2*(int(n**0.5)!=(n**0.5))-1) for n in range(left,right+1))
