def solution(n):
    answer = 0
    col_pos = [0] * n
    diag1 = [0] * (2 * n - 1)
    diag2 = [0] * (2 * n - 1)

    def backtrack(row):
        nonlocal answer
        if row == n:
            answer += 1
            return
        for col in range(n):
            if col_pos[col] or diag1[row + col] or diag2[row - col + n - 1]:
                continue
            col_pos[col] = diag1[row + col] = diag2[row - col + n - 1] = 1
            backtrack(row + 1)
            col_pos[col] = diag1[row + col] = diag2[row - col + n - 1] = 0

    backtrack(0)
    return answer
