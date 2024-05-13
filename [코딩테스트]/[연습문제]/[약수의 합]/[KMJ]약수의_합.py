def solution(n):
    answer = 0
    for i in range(1, int(n**(1/2)) + 1):
        if (n % i == 0):
            answer += i 
            if ( (i**2) != n) : 
                answer += (n // i)
    return answer
