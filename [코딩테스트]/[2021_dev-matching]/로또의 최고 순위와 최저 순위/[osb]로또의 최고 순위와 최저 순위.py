def score_board(num):
    if num == 0:
        return 6
    else:
        return 7-num

def solution(lottos, win_nums):
    max_score, min_score = 0, 0
    answer = []
    for num in lottos:
        if num in win_nums:
            max_score += 1
            min_score += 1
        elif num == 0:
            max_score +=1
    answer.append(score_board(max_score))
    answer.append(score_board(min_score))
    return answer