def solution(m, n, startX, startY, balls):
    answer = []
    for [x,y] in balls:
        D = []
        S = [startX,startY]
        if not(startX>x and startY==y):
            Z = [-x,y]
            D.append((S[0]-Z[0])**2 + (S[1]-Z[1])**2)
        if not(startX<x and startY==y):
            Z = [2*m-x,y]
            D.append((S[0]-Z[0])**2 + (S[1]-Z[1])**2)
        if not(startX==x and startY>y):
            Z = [x,-y]
            D.append((S[0]-Z[0])**2 + (S[1]-Z[1])**2)
        if not(startX==x and startY<y):
            Z = [x,2*n-y]
            D.append((S[0]-Z[0])**2 + (S[1]-Z[1])**2)
        answer.append(min(D))
    return answer
