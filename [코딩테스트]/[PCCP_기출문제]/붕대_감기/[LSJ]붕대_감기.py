def solution(bandage, health, attacks):
    answer = 0
    initial_health = health
    attack_time = [attack[0] for attack in attacks]
    attack_damage = [attack[1] for attack in attacks]

    for i in range(len(attack_time)):
        if i > 0:
            time_diff = attack_time[i] - attack_time[i-1]
            if time_diff >= 2:
                health += (time_diff - 1) * bandage[1]

            while time_diff > bandage[0]:
                health += bandage[2]
                time_diff -= bandage[0]

            if health > initial_health:
                health = initial_health

        health -= attack_damage[i]
        if health <= 0:
            answer = -1
            break

    if health > 0:
        answer = health

    return answer
