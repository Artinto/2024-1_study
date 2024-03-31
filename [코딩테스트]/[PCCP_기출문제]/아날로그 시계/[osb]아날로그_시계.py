def solution(h1, m1, s1, h2, m2, s2):
    time = 3600 * (h2 - h1) + 60 * (m2 - m1) + (s2 - s1)
    hour_angle = (3600 * h1 + 60 * m1 + s1) % (360*120)
    min_angle = ((60 * m1 + s1) *12) % (360*120)
    sec_angle = (s1 * 720) % (360*120)
    answer = 0
    elapsed_time = 1
    #print(hour_angle, min_angle, sec_angle)
    def hour_check(hour, sec, next_hour, next_sec):
        if ((sec < hour) and (next_sec > next_hour)):
            return True
        if (next_sec == 0 and next_hour != 0) and ((360*120) > next_hour) and (sec < hour):
            return True
        return False

    def min_check(min, sec, next_min, next_sec):
        if (sec < min) and (next_sec > next_min):
            return True
        if (next_sec == 0 and next_min != 0) and ((360*120) > next_min) and (sec < min):
            return True
        return False
    pre_hour, pre_min, pre_sec = hour_angle, min_angle, sec_angle
    while elapsed_time <= time:
        next_hour = (hour_angle + elapsed_time) % (360*120)
        next_min = (min_angle + elapsed_time*12) % (360*120)
        next_sec = (sec_angle + elapsed_time * 720) % (360*120)
        if hour_check(pre_hour, pre_sec, next_hour, next_sec):
            answer += 1
        if min_check(pre_min, pre_sec, next_min, next_sec):
            answer += 1
        if pre_sec == pre_hour or pre_min==pre_sec:
            answer += 1
        pre_hour = next_hour
        pre_min = next_min
        pre_sec = next_sec
        elapsed_time += 1

    return answer
