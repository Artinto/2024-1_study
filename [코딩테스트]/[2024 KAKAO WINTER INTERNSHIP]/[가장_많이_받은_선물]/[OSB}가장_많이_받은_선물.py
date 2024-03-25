from collections import defaultdict
def solution(friends, gifts):
    # 선물 그래프
    give_gift = defaultdict(dict)
    receive_gift = defaultdict(dict)
    for gift in gifts:
        giver, receiver = gift.split()
        if receiver not in give_gift[giver]:
            give_gift[giver][receiver] = 0
        give_gift[giver][receiver] += 1
        
        if giver not in receive_gift[receiver]:
            receive_gift[receiver][giver] = 0
        receive_gift[receiver][giver] += 1
    # 친구별 계산
    gift_indices = defaultdict(int)
    for friend in friends:
        given = sum(give_gift[friend].values())
        received = sum(receive_gift[friend].values())
        gift_indices[friend] = given - received
    next_month_gifts = defaultdict(int)
    for i in range(len(friends)):
        for j in range(i+1, len(friends)):
            friend1, friend2 = friends[i], friends[j]
            # 두 사람 간에 선물 수 비교
            if give_gift[friend1].get(friend2, 0) < give_gift[friend2].get(friend1, 0): # get에서 return할께 없다면 0으로
                next_month_gifts[friend2] += 1
            elif give_gift[friend1].get(friend2, 0) > give_gift[friend2].get(friend1, 0):
                next_month_gifts[friend1] += 1
            else:
                if gift_indices[friend1] < gift_indices[friend2]:
                    next_month_gifts[friend2] += 1
                elif gift_indices[friend1] > gift_indices[friend2]:
                    next_month_gifts[friend1] += 1
    if next_month_gifts: # 주고 받을게 있는지 확인.
        return max(next_month_gifts.values()) 
    else:
        return 0