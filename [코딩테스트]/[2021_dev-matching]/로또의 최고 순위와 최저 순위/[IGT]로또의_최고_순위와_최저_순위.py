def solution(lottos, win_nums):
    win_count = len(set(lottos) & set(win_nums))
    return [min(6, 7-(win_count + lottos.count(0))), min(6, 7-win_count)]
