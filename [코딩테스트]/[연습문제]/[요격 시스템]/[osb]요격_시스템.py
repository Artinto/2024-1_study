def solution(targets):
    # 끝나는 위치, 시작 위치 순으로 정렬.
    targets.sort(key=lambda x: (x[1], x[0]))  
   
    answer = 1
    last_end_pos = targets[0][1]

    for start_pos, end_pos in targets[1:]:
        # 미사일 요격이 필요해지는 경우.
        if start_pos >= last_end_pos:
            answer += 1
            last_end_pos = end_pos

    return answer