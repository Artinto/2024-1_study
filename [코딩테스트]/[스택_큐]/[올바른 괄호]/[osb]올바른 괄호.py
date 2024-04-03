from collections import deque
def solution(s):
    dq = deque()
    for i in s:
        if i == '(':
            dq.append('(')
        else:
            if len(dq) == 0:
                return False
            elif dq.pop() != '(':
                return False
            
    if len(dq) == 0:
        answer = True
    else:
        answer = False
    return answer