def solution(plans):
    answer = []
    for i in range(len(plans)):
        h, m = map(int, plans[i][1].split(':')) # 시와 분 분할
        plans[i][1] = h*60+m # 분 단위로 재구성
        plans[i][2] = int(plans[i][2])
        
    plans.sort(key=lambda x:x[1]) # 과제 시작 순서대로 정렬
    stack = [] # 끝내지 못한 과제를 저장할 stack
    
    for i in range(len(plans)):
        if i == len(plans)-1: # 마지막 과제라면 그냥 stack에 넣어 빠르게 계산
            stack.append(plans[i])
            break
        
        sub, st, t = plans[i]
        nsub, nst, nt = plans[i+1]
        if st + t <= nst: #시간 내에 과제 완료
            answer.append(sub)
            temp_time = nst - (st+t) # 남은 시간 저장
            
            while temp_time != 0 and stack: # 남은 시간이 있고, stack이 있으면
                tsub, tst, tt = stack.pop() # 가장 최근에 중단된 과제
                if temp_time >= tt: # 과제 완료 가능
                    answer.append(tsub)
                    temp_time -= tt # 과제한 만큼 남은 시간 삭제
                else:
                    stack.append([tsub, tst, tt - temp_time]) # 남은 시간만큼 과제하고 다시 stack에 저장
                    temp_time = 0
            
        else: # 시간 내에 과제 완료 못하면
            plans[i][2] = t - (nst - st) # 남은 과제 시간 초기화
            stack.append(plans[i]) # stack에 저장
        
    while stack: # 남은 시간이 없고, stack은 남아있으면
        sub, st, tt = stack.pop()
        answer.append(sub) # 최근 순서대로 과제 진행

    return answer
