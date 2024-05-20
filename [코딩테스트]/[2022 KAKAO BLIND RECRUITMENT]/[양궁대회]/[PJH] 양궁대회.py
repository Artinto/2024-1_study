def solution(n, info):
    answer = [-1]
    max_diff = 0

    def dfs(idx, arrows, lion_info):
        nonlocal max_diff, answer
        if idx > 10 or arrows == 0:
            lion_info[10] += arrows
            lion_score = 0
            apeach_score = 0
            for i in range(11):
                if lion_info[i] == 0 and info[i] == 0:
                    continue
                if lion_info[i] > info[i]:
                    lion_score += 10 - i
                else:
                    apeach_score += 10 - i
            diff = lion_score - apeach_score
            if diff > 0 and diff > max_diff:
                max_diff = diff
                answer = lion_info[:]
            elif diff > 0 and diff == max_diff:
                for i in range(10, -1, -1):
                    if answer[i] > lion_info[i]:
                        break
                    elif answer[i] < lion_info[i]:
                        answer = lion_info[:]
                        break
            return

        if arrows > info[idx]:
            new_lion_info = lion_info[:]
            new_lion_info[idx] = info[idx] + 1
            dfs(idx + 1, arrows - info[idx] - 1, new_lion_info)
        dfs(idx + 1, arrows, lion_info)

    dfs(0, n, [0] * 11)
    return answer
