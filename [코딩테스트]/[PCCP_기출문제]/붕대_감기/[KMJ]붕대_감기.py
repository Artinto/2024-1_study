def solution(bandage, health, attacks):
    health_max = health
    count = 0
    time = attacks[-1][0]
    for i in range(1,time+1):
        if i == attacks[0][0]:
            health -= attacks[0][1]
            del attacks[0]
            count = 0
            
            if health <= 0:
                return -1
        else:
            count += 1
            if count < bandage[0]:
                health += bandage[1]    
            else:
                health += (bandage[1] + bandage[2])
                count = 0
            if health <= 0:
                return -1
        if health > health_max:
            health = health_max   
    return health
