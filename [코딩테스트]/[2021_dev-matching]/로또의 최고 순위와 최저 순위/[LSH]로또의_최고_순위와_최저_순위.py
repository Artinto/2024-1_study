def solution(lottos, win_nums):
    answer = []
    ranking = {6:1,  5:2, 4:3, 3:4, 2:5} # 맞춘 개수에 따른 등수 설정
    match_count = sum(1 for num in lottos if num in win_nums and num !=0) # 지워진 숫자를 제외하고 맞춘 개수
    zeros_count = lottos.count(0) # 지워진 숫자의 개수
    
    return [ranking.get(match_count+zeros_count,6),ranking.get(match_count,6)] # [최고 순우,최저 순위]
