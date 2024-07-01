# x, y 를 r1~r2로 지정
# case 1) sqrt(x^2 + y^2)가 r1보다 같거나 크고, r2보다 같거나 작으면 두 원 사이의 정수
# case 2) x축과 y축 사이의 점 (2번 계산됨)
# (case 1 - case 2) * 4

# 런타임 에러
# def solution(r1, r2):
#     answer = 0
#     x, y = 0, 0
#     case1 = 0
#     for x in range(0, r2+1):
#         for y in range(0, r2+1):
#             if r1 <= (x**2 + y**2)**0.5 <= r2:
#                 case1 += 1
#             if (x**2 + y**2)**0.5 > r2: # 빠른 계산을 위해 대각선이 r2보다 크면 for문 통과
#                 break
#     case2 = (r2-r1 + 1)
#     answer = (case1 - case2)*4
#     return answer


def solution(r1, r2):
    case1 = 0
    # x를 기준으로 두 원 사이에 있는 원의 개수
    for x in range(1, r2+1): # x축과 y축의 점을 모두 계산하면 점이 겹치므로 x축은 제외하고 x=1부터 계산
        
        if x < r1: # r1 내부에 있는 점
            y_min = (r1**2 - x**2 - 1) ** 0.5 # 예외로 인해 -1 (5^2 - 3^2 = 4^2)
        else:
            y_min = 0 # -1을하면 음수가 되어 루트를 못하는 경우가 생겨서 예외 제거하기 위해 0
        
        y_max = (r2**2 - x**2) ** 0.5  # r2 내부에 있는 점
        
        
        case1 += int(y_max) - int(y_min) # r2 내부 - r1 내부 중 정수인 점
    
    case2 = r2 - r1 + 1 # x축 위 점 개수
    
    answer = (case1 + case2) * 4
    return answer
