def solution(price, money, count):
    total_price = 0
    for ride in range(1,count+1): # 1~count 까지
        total_price += ride * price
    
    answer = total_price - money
    if answer <= 0:
        return 0
    return answer