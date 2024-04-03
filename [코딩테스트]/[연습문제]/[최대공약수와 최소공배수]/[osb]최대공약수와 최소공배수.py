def solution(n, m):
    # 일단 큰 수 먼저 찾기.
    if n >= m:
        num1, num2 = n, m
    else:
        num1, num2 = m, n
    
    for num in range(1, num1//2+1): # 큰 수의 1/2 제곱근까지 진행했을때의 가장 크게 나눠지는게 최대 공배수
        if num1 % num == 0 and num2 % num == 0:
            max_commom_divisor = num
    least_common_multiple = num1 * num2 // max_commom_divisor # num1 * num2를 최대 공배수로 나눈게 최소공배수.
    answer = [max_commom_divisor, least_common_multiple]
    return answer