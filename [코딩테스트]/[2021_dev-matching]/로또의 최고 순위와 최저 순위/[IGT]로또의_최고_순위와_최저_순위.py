def solution(lottos, win_nums):
    win_num = len(set(lottos) & set(win_nums))
    return [min(6, 7-(win_num + lottos.count(0))), min(6, 7-win_num)]
