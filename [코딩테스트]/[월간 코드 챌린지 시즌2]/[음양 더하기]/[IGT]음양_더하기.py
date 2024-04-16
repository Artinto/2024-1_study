def solution(absolutes, signs):
    return sum(absolute*(2*sign-1) for absolute,sign in zip(absolutes,signs))
