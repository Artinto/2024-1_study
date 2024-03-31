def solution(bandage, health, attacks):
    current_health = health
    heal_time, heal, bonus_heal = bandage[0], bandage[1], bandage[2]
    temp = 0
    for attack in attacks:
        time, demage = attack[0], attack[1]
        # 처음 체력회복
        healtime = time - temp-1
        check_bouns_heal = healtime // heal_time
        current_health += (healtime * heal + check_bouns_heal*bonus_heal)
        if current_health >= health:
            current_health = health
        current_health -= demage
        temp = time
        if current_health <= 0:
            break
    if current_health <= 0:
        return -1
    return current_health