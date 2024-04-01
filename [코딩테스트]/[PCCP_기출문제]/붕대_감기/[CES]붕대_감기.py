# bandage = [시전 시간, 초당 회복량, 추가 회복량]
# health = [최대 체력]
# attacks = [공격시간, 피해량]

def solution(bandage, health, attacks):
    max_health = health
    last_attack_time = attacks[-1][0]
    attack_dict = {}
    for i in attacks:
        attack_dict[i[0]] = i[1]
    t = 0
    continue_sec = 0
    
    while t <= last_attack_time:
        
        if t in attack_dict:
            health -= attack_dict[t]
            continue_sec = 0
            
            if health <= 0:
                return -1
        else:
            continue_sec += 1
            if continue_sec < bandage[0]:
                health += bandage[1]
                if health > max_health:
                    health = max_health
            else:
                health = health + bandage[1] + bandage[2]
                if health > max_health:
                    health = max_health
                continue_sec = 0
        t += 1
    return health
