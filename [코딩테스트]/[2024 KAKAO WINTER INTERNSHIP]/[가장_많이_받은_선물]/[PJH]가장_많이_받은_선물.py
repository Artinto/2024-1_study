def solution(friends, gifts):
    answer = 0
    
    # 초기화
    present = {}
    pre_idx = {}
    for friend in friends:
        present[friend] = {}
        pre_idx[friend] = 0
        
    for gift in gifts:
        g, r = gift.split(" ")
        if r in present[g]:
            present[g][r] += 1
        else:
            present[g][r] = 1
        pre_idx[g] += 1
        pre_idx[r] -= 1
        
    get_count = [0 for _ in friends]
    for i in range(len(friends)):
        buddy = friends[i]
        for j in range(i+1, len(friends)):
            another = friends[j]
            A = present[buddy][another] if another in present[buddy] else 0
            B = present[another][buddy] if buddy in present[another] else 0
            
            if A > B:
                get_count[i] += 1
            elif B > A:
                get_count[j] += 1
            elif A == B:
                A_idx, B_idx = pre_idx[buddy], pre_idx[another]
                if A_idx > B_idx:
                    get_count[i] += 1
                elif B_idx > A_idx:
                    get_count[j] += 1
    answer = max(get_count)
    return answer

