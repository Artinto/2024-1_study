def solution(s):
    answer = 0
    for i in s:
        answer += -1 + 2*(i=='(')
        if answer < 0:
            return False
    return False if answer else True
