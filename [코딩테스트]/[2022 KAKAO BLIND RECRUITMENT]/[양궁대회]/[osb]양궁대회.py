from itertools import combinations_with_replacement
from collections import Counter
def solution(n, info):
    max_diff = 0
    answer = [0] * 11
    # 브루트 포스로 모든 경우의 수 확인하기
    for lion in combinations_with_replacement(range(11), n):
        lion_info = Counter(10 - score for score in lion)
        apeach_score = 0
        lion_score = 0
        
        for i in range(11):
            if info[i] == lion_info[i] == 0:
                continue
            if info[i] >= lion_info[i]:
                apeach_score += 10 - i
            else:
                lion_score += 10 - i
        
        diff = lion_score - apeach_score
        if diff > max_diff:
            max_diff = diff
            answer = [lion_info[i] for i in range(11)]
    
    if max_diff == 0:
        return [-1]
    else:
        return answer