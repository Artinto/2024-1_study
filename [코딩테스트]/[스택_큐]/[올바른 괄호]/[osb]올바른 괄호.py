def solution(s):
    stack = []
    for i in s:
        if i == '(':
            stack.append('(')
        else:
            if len(stack) == 0:
                return False
            elif stack.pop() != '(':
                return False
    return len(stack) == 0