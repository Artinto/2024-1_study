def solution(m, n, startX, startY, balls):
    answer = []
    x1, y1 = startX, -1*startY
    x2, y2 = -1*startX, startY
    x3, y3 = m+(m-startX), startY
    x4, y4 = startX, n+(n-startY)
    for ball in balls:
        x, y = ball
        length_1= (x-x1)**2 + (y-y1)**2
        length_2= (x-x2)**2 + (y-y2)**2
        length_3= (x-x3)**2 + (y-y3)**2
        length_4= (x-x4)**2 + (y-y4)**2
        if y == startY:
            if startX < x < x3:
                length_3 = 1e9
            elif x2 < x < startX:
                length_2 = 1e9
        elif x == startX:
            if startY < y < y4:
                length_4 = 1e9
            elif y1 < y < startY:
                length_1 = 1e9
        answer.append(min(length_1, length_2, length_3, length_4))
    return answer