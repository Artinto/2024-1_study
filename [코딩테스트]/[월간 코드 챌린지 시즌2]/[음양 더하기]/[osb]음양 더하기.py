def solution(left, right):
    answer = 0
    for num in range(left, right+1):
        if num%(num**0.5)==0:
            answer -= num
        else:
            answer += num
    return answer