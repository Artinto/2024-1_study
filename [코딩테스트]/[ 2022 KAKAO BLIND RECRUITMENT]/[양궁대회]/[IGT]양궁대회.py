def solution(n, info_A):
    score_A = sum((i>0)*(10-idx) for idx,i in enumerate(info_A))
    info_L = [0]*11
    arrow_L = [i+1 for i in info_A[:-1]]
    rate = [[10-idx,(2-(hit==1))*(10-idx),hit] for idx,hit in enumerate(arrow_L)]
    rate.sort(key=lambda x:x[0])
    dp = [[0,[],11] for _ in range(n+1)]
    for a,b,c in rate:
        for x in range(n,c-1,-1):
            if dp[x-c][0]+b > dp[x][0] or (dp[x-c][0]+b == dp[x][0] and dp[x-c][2] < dp[x][2]):
                dp[x][0] = dp[x-c][0]+b
                dp[x][1] = dp[x-c][1]+[a]
                dp[x][2] = min(dp[x][1])
    for i in dp[n][1]:
        info_L[10-i] = arrow_L[10-i]
    info_L[-1] = n - sum(info_L)
    answer = dp[n][0] - score_A
    return info_L if answer>0 else [-1]
