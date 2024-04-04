def solution(lottos, win_nums):
    answer = []
    ranking = {6:1,  5:2, 4:3, 3:4, 2:5}
    match_count = sum(1 for num in lottos if num in win_nums and num !=0)
    zeros_count = lottos.count(0)
    
    return [ranking.get(match_count+zeros_count,6),ranking.get(match_count,6)]
