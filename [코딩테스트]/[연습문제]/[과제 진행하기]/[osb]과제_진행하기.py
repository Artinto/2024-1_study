def time_to_minutes(time):
        hour, minute = map(int, time.split(':'))
        return 60 * hour + minute

def solution(plans):
    plans = sorted(plans, key = lambda x: x[1]) # nlogn
    remain_list = []
    answer = []
    for idx, plan in enumerate(plans[:-1]):
        study, start, time = plan
        current_time = time_to_minutes(start)
        next_time = time_to_minutes(plans[idx + 1][1])
        remain_time = int(time) - (next_time - current_time)
        if remain_time == 0:
            answer.append(study)
        elif remain_time > 0:
            remain_list.append((study, remain_time))
        else:
            answer.append(study)
            remain_time = -remain_time
            while remain_time > 0 and remain_list:
                remain_study, study_time = remain_list.pop()
                if study_time <= remain_time:
                    answer.append(remain_study)
                    remain_time -= study_time
                else:
                    remain_list.append((remain_study, study_time - remain_time))
                    remain_time = 0
    # 가장 마지막 추가
    answer.append(plans[-1][0])
    answer.extend(study for study, _ in remain_list[::-1])
    return answer