def solution(line):
    answer = []
    cross = set() # 교차점을 저장
    
    for i in range(len(line)):
        # 선의 방정식 i
        a, b, e = line[i]
        for j in range(i + 1, len(line)):
            # 선의 방정식 j
            c, d, f = line[j]
            
            if (a * d) - (b * c) != 0:
                # (a * d) - (b * c)가 0이면 두 직선은 평행 또는 일치
                # x, y 교차점을 구하는 규칙
                x = (b * f - e * d) / (a * d - b * c)
                y = (e * c - a * f) / (a * d - b * c)
            
            if int(x) == x and int(y) == y:
                # 정수 형태로 나타나면 교차점
                x = int(x) # float 형태 -> int
                y = int(y)
                cross.add((x, y)) # 교차점 추가
                
    # 그리드의 크기를 확인하기 위해 min, max 값 구하기
    min_x = min(point[0] for point in cross)
    max_x = max(point[0] for point in cross)
    min_y = min(point[1] for point in cross)
    max_y = max(point[1] for point in cross)
    
    
    for y in range(max_y, min_y - 1, -1):
        # 역순으로 y좌표 (y축은 큰게 먼저 작성)
        row = ""
        for x in range(min_x, max_x + 1):
            # x축은 작은게 먼저 작성 (왼쪽)
            if (x, y) in cross:
                # x,y가 교차점 리스트에 있으면
                row += "*"
                # 별 작성
            else:
                # 교차점 리스트에 없으면
                row += "."
                # . 작성
        answer.append(row)
                    
    return answer
