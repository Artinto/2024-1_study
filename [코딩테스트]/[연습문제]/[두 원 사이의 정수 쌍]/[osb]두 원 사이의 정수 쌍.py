def count_points_inside_circle(r):
    count = 0
    end_point = 0
    for x in range(1, r+1):
        max_y = int((r**2 - x**2)**0.5)
        count+=max_y*4
        if max_y **2 == r**2 - x**2: # 작은 원의 경계에 있을 경우
            end_point +=1
    return count + 4*r +1, 4*end_point

def solution(r1, r2):
    answer = 0
    if r1> r2 :
        big_r = r1
        small_r = r2
    else:
        big_r = r2
        small_r = r1
    big_count, end_big = count_points_inside_circle(big_r)
    small_count, end_small = count_points_inside_circle(small_r)
    answer = big_count - small_count + end_small
    return answer