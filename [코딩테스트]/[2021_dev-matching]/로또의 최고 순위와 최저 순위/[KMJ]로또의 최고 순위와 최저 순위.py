def solution(lottos, win_nums):
    zero = 0
    count = 0
    for i in lottos:
        if i == 0:
            zero += 1
        if i in win_nums:
            count += 1
    return [min(7-(count+zero),6), min(7-count,6)]
