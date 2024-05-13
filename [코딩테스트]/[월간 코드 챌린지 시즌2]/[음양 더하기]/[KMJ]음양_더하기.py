def solution(absolutes, signs):
    answer = 0
    for i in range(len(signs)):
        if signs[i] == False:
            absolutes[i] = -1 * absolutes[i]
        answer += absolutes[i]
    return answer
