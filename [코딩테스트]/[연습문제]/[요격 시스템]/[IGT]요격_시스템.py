def solution(targets):
    targets.sort(key=lambda x:x[1])
    answer = 0
    missile = -1
    for i in targets:
        if i[0] >= missile:
            missile = i[1]
            answer += 1
    return answer
