def solution(left, right):
    answer = 0
    for num in range(left, right+1):
        point = 0
        for i in range(1, int(num**(1/2)) + 1):
            if (num % i == 0):
                point += 1 
                if ( (i**2) != num) : 
                    point += 1
        if point % 2 == 0:
            answer += num
        else:
            answer -= num
    return answer

___

def solution(left, right):
    answer = 0
    for num in range(left, right+1):
        if num % num**(1/2) == 0:
            answer -= num
        else:
            answer += num
    return answer
