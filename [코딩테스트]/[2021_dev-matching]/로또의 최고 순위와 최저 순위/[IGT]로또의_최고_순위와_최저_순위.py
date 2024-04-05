def solution(lottos, win_nums):
    win_num = len(set(lottos) & set(win_nums))
    max_win_num = win_num + lottos.count(0)
    return [min(6,7-max_win_num),min(6,7-win_num)]
