def solution(bandage, max_health, attacks):
    now, health = attacks[0][0], max_health - attacks[0][1]
    for time,damage in attacks[1:]:
        t = time - now - 1
        health = min(max_health, health + t*bandage[1] + (t//bandage[0])*bandage[2])
        now = time
        health -= damage
        if health <= 0:
            return -1
    return health
