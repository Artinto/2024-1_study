def solution(price, money, count):
    return max(((((count*(count+1))/2)*price) - money), 0)
