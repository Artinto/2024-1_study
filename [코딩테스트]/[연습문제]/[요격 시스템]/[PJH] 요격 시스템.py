def solution(targets):
    targets.sort(key=lambda x: x[1])
    
    intercepts = 0
    current_end = -1
    
    for s, e in targets:
        if s >= current_end:
            intercepts += 1
            current_end = e
    answer = intercepts
    return answer

