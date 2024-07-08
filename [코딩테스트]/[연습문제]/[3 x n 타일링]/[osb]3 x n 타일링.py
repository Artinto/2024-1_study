def solution(n):
    dp = [0] * 5001
    dp[2] = 3
    dp[4] = 11
    for idx in range(6, 5001,2):
        dp[idx] = 3*dp[idx-2]+2
        for i in range(2, idx-3, 2):
            dp[idx] += 2*dp[i]
        dp[idx] = dp[idx] % 1000000007
    return dp[n]