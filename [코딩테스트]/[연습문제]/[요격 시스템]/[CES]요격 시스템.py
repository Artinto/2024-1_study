def solution(targets):
    targets.sort(key = lambda x: x[0])
    # 각 구간의 시작점을 기준으로 오름차순으로 정렬 -> 구간이 시작하는 순서대로 정렬
    end = targets[0][1]
    # 첫 번째 구간의 종료점 저장
    answer = 1
    for s,e in targets[1:]:
        if s >= end:
            answer += 1
            # 새로운 구간이 선택되었음을 의미
            end = e
            # 종료점 업데이트
        else:
            if e < end:
                # 현재 구간이 이전 구간에 완전히 포함될 경우
                end = e
    return answer
