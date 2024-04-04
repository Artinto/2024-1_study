def solution(s):
    dic = {'(':1,')':-1} # 문자 별 값 설정
    total = 0
    s_list = list(s)
    for char in s_list:
        total += dic[char] # 반복문 동안
        if total < 0: # total이 0 아래로 떨어지면 
            return False # 올바른 짝이 아님
    if total != 0: # 마지막이 0이 아닐때도
        return False # 개수가 맞지 않는 것
    return True #반복문이 다 돌면 짝 맞는것
