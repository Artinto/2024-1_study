from itertools import combinations
def solution(line):
    temp = []
    combis = combinations(line, 2)
    meet_points = []
    for combi in combis:
        line1, line2 = combi
        a, b, e = line1
        c, d, f = line2
        # 교점이 없는 경우
        if a*d - b*c == 0:
            continue
        x = (b*f - e*d) / (a*d - b*c)
        y = (e*c - a*f) / (a*d - b*c)
        if x == int(x) and y == int(y): # 정수인 경우만 추가
            meet_points.append((int(x), int(y)))
    min_x = min(meet_points, key=lambda x: x[0])[0]
    max_x = max(meet_points, key=lambda x: x[0])[0]
    min_y = min(meet_points, key=lambda x: x[1])[1]
    max_y = max(meet_points, key=lambda x: x[1])[1]
    for i in range(min_y, max_y+1):
        row = ['.'] * (max_x - min_x + 1)
        temp.append(row)
    for meet_point in meet_points:
        x, y = meet_point
        temp[max_y - y][x - min_x] = '*' #y좌표는 뒤집어서 계산
    answer = []
    for row in temp:
        answer.append(''.join(row))
    return answer