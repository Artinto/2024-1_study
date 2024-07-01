import math

def solution(plans):
    answer = []

    # 'hh:mm' 형식의 시간을 분 단위로 변환하는 함수
    def time_to_minutes(time):
        hours, minutes = map(int, time.split(':'))
        return hours * 60 + minutes

    # 계획을 분 단위 시작 시간으로 변환하고, 시작 시간 순으로 정렬
    tasks = [(name, time_to_minutes(start), int(playtime)) for name, start, playtime in plans]
    tasks.sort(key=lambda x: x[1])
    
    result = []
    stack = []
    current_time = 0  # 현재 시간을 0으로 초기화
    
    # 각 작업을 순차적으로 처리
    for name, start, playtime in tasks:
        # 새로운 작업을 시작하기 전에 현재 진행 중인 작업 처리
        while stack and current_time < start:
            ongoing_task, remaining_time, start_time = stack.pop()
            if current_time + remaining_time <= start:
                # 현재 진행 중인 작업을 완료할 수 있는 경우
                current_time += remaining_time
                result.append(ongoing_task)
            else:
                # 현재 진행 중인 작업을 완료할 수 없는 경우
                remaining_time -= (start - current_time)
                stack.append((ongoing_task, remaining_time, start_time))
                current_time = start
        
        # 새로운 작업을 시작
        stack.append((name, playtime, start))
        current_time = start
    
    # 남아 있는 작업들을 처리
    while stack:
        task_name, remaining_time, start_time = stack.pop()
        result.append(task_name)
    
    answer = result  # 결과를 answer에 저장
    return answer
