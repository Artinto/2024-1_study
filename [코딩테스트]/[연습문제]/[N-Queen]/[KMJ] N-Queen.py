def solution(n):
    rows = set()  # 퀸이 놓인 열
    diag1 = set()  # '/' 방향의 대각선 (col - row)
    diag2 = set()  # '\' 방향의 대각선 (col + row)
    
    def backtrack(col):
        if col == n: # 모든 퀸이 놓였으므로 가능한 배치임을 설명
            return 1
        
        count = 0  # 유효한 배치의 수를 세기 위한 변수 초기화
        for row in range(n):  # 각 열에 퀸을 놓을 수 있는지 확인
            # 이미 퀸이 놓인 열, '/' 대각선에 퀸이 놓인 열, '\' 대각선에 퀸이 놓인 열
            if row in rows or (col - row) in diag1 or (col + row) in diag2:
                continue  # 다음 행으로 이동
            
            # 해당 열에 퀸을 놓을 수 있다면
            rows.add(row)
            diag1.add(col - row)
            diag2.add(col + row)
            
            # 다음 열에서 가능한 행 탐색, count에 가능한 경우의 수 더하기
            count += backtrack(col + 1)
            
            # 백트래킹: 퀸을 제거하고 집합에서 삭제
            rows.remove(row)
            diag1.remove(col - row)
            diag2.remove(col + row)
        
        return count  # 유효한 배치의 수를 반환
    
    return backtrack(0)  # 0번째 열부터 시작하여 모든 유효한 배치의 수를 반환
